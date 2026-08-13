"""Relevance-based metadata retrieval.

Rather than sending the full schema to the LLM on every call, this module
scores each table/column against the user's question (and any table hints
the intent-understanding node produced) and returns only the slice of
metadata that's actually relevant -- plus anything reachable from it via a
foreign key, so joins remain possible. This is what keeps the system
scalable to schemas far larger than the three tables shipped here.

Two table-selection strategies are available:

* **Lexical** (:func:`select_relevant_tables`) -- pure keyword/token overlap
  against table/column names, descriptions, and sample values. Zero setup,
  deterministic, and the fallback whenever no vector index is available.
* **Vector/RAG** (:func:`select_relevant_tables_rag`) -- nearest-neighbor
  search over table documents embedded in :mod:`app.metadata.vector_store`.
  This generalizes to paraphrased questions that share no literal keywords
  with the schema (e.g. "top sellers" matching a table whose description
  says "product sales facts"), at no LLM token cost. It's tried first
  whenever a ``db_identity`` is supplied, and falls back to the lexical
  scorer whenever the vector store has nothing indexed yet or errors.

Either way, the *output* -- a trimmed metadata dict rendered to prompt text
via :func:`format_metadata_for_prompt` -- is identical, so callers/tests that
don't care about retrieval strategy can ignore the distinction entirely.
"""

from __future__ import annotations

import re
from typing import Any

from app.metadata import vector_store

_STOPWORDS = frozenset(
    {
        "a", "an", "the", "of", "for", "and", "or", "by", "to", "in", "on",
        "what", "which", "who", "how", "many", "much", "is", "are", "was",
        "were", "show", "me", "list", "get", "find", "give", "please",
        "with", "per", "top", "all", "each", "than", "vs", "versus",
        "over", "last", "this", "that", "our",
    }
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9]*")

# Keep prompts bounded on wider schemas. Foreign-key neighbors needed for joins
# are added after ranking and are allowed to exceed this soft relevance budget.
MAX_RELEVANT_TABLES = 8
MAX_COLUMNS_PER_TABLE = 80
# Chroma's default cosine-style distance is lower-is-better. Hits beyond this
# are treated as guesses so a large unseen schema is catalog-disambiguated.
MAX_VECTOR_DISTANCE = 1.25

_COLUMN_NAME_WEIGHT = 3
_TABLE_NAME_WEIGHT = 4
_DESCRIPTION_WEIGHT = 1
_SAMPLE_VALUE_WEIGHT = 2


def _tokenize(text: str) -> set[str]:
    # Natural-language "sales amount" should match identifiers such as
    # SalesAmount; split lower-to-upper camel-case boundaries before tokenizing.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(text):
        word = raw.lower()
        if len(word) <= 1 or word in _STOPWORDS:
            continue
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        tokens.add(word)
    return tokens


def _score_table(table_name: str, table_meta: dict[str, Any], question_tokens: set[str]) -> int:
    score = 0
    name_tokens = _tokenize(table_name)
    score += _TABLE_NAME_WEIGHT * len(name_tokens & question_tokens)
    score += _DESCRIPTION_WEIGHT * len(
        _tokenize(table_meta.get("description", "")) & question_tokens
    )

    for col_name, col_meta in table_meta.get("columns", {}).items():
        col_tokens = _tokenize(col_name)
        score += _COLUMN_NAME_WEIGHT * len(col_tokens & question_tokens)
        score += _DESCRIPTION_WEIGHT * len(
            _tokenize(col_meta.get("description", "")) & question_tokens
        )
        for sample in col_meta.get("sample_values") or []:
            score += _SAMPLE_VALUE_WEIGHT * len(
                _tokenize(str(sample)) & question_tokens
            )

    return score


def has_lexical_schema_signal(metadata: dict[str, Any], question: str) -> bool:
    """Whether names/descriptions/samples provide any deterministic match."""
    tables = metadata.get("tables", {})
    if _explicit_table_mentions(tables, question):
        return True
    tokens = _tokenize(question)
    return any(_score_table(name, table, tokens) > 0 for name, table in tables.items())


def _related_tables(metadata: dict[str, Any], table_names: set[str]) -> set[str]:
    related = set()
    for rel in metadata.get("relationships", []):
        if rel["from_table"] in table_names:
            related.add(rel["to_table"])
        if rel["to_table"] in table_names:
            related.add(rel["from_table"])
    return related


def _explicit_table_mentions(
    tables: dict[str, Any], question: str
) -> list[str]:
    """Return catalog table names written explicitly in the question.

    An exact table reference is stronger than a semantic search result.  It
    also avoids embedding lookup and automatic FK-neighbour expansion for a
    self-contained question that names a catalog table. If the user names
    multiple tables they are all retained, so explicit join requests still
    receive the required schemas.
    """
    mentioned: list[tuple[int, str]] = []
    for name in tables:
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
            question,
            flags=re.IGNORECASE,
        )
        if match:
            mentioned.append((match.start(), name))
    return [name for _position, name in sorted(mentioned)]


def _expand_explicit_for_missing_concepts(
    metadata: dict[str, Any], question: str, explicit: list[str]
) -> list[str]:
    """Add only neighbors that explain concepts absent from named tables."""
    tables = metadata.get("tables", {})
    covered: set[str] = set()
    for name in explicit:
        table = tables[name]
        covered |= _tokenize(name)
        covered |= _tokenize(table.get("description", ""))
        for column_name, column in table.get("columns", {}).items():
            covered |= _tokenize(column_name)
            covered |= _tokenize(column.get("description", ""))
    missing = _tokenize(question) - covered
    if not missing:
        return explicit
    neighbors = _related_tables(metadata, set(explicit)) - set(explicit)
    additions = sorted(
        (
            name for name in neighbors
            if _score_table(name, tables.get(name, {}), missing) > 0
        ),
        key=lambda name: -_score_table(name, tables.get(name, {}), missing),
    )
    return [*explicit, *additions]


def select_relevant_tables(
    metadata: dict[str, Any],
    question: str,
    hinted_tables: list[str] | None = None,
) -> list[str]:
    """Pick the tables relevant to ``question``, expanded with FK neighbors."""
    tables = metadata.get("tables", {})
    explicit = _explicit_table_mentions(tables, question)
    if explicit:
        return _expand_explicit_for_missing_concepts(metadata, question, explicit)
    question_tokens = _tokenize(question)

    scores = {
        name: _score_table(name, meta, question_tokens) for name, meta in tables.items()
    }

    ranked = sorted(tables, key=lambda name: (-scores[name], name.casefold()))
    selected = set(
        [name for name in ranked if scores[name] > 0][:MAX_RELEVANT_TABLES]
    )

    names_by_lower = {name.casefold(): name for name in tables}
    for hint in hinted_tables or []:
        canonical = names_by_lower.get(hint.casefold())
        if canonical:
            selected.add(canonical)

    # Nothing matched lexically (short/ambiguous question) -- fall back to
    # the full catalog rather than guessing wrong and failing SQL generation.
    if not selected:
        selected = set(ranked[:MAX_RELEVANT_TABLES])

    selected |= _related_tables(metadata, selected)

    return sorted(selected, key=lambda n: -scores.get(n, 0))


def select_relevant_tables_rag(
    metadata: dict[str, Any],
    question: str,
    hinted_tables: list[str] | None = None,
    *,
    db_identity: str | None = None,
    top_k: int | None = None,
) -> tuple[list[str], str]:
    """Pick relevant tables via vector search, falling back to lexical scoring.

    Returns ``(table_names, mode)`` where ``mode`` is ``"vector"`` or
    ``"lexical"`` so callers can surface which strategy actually answered
    the question (see the UI's retrieval-mode badge).
    """
    tables = metadata.get("tables", {})
    explicit = _explicit_table_mentions(tables, question)
    if explicit:
        # "lexical" remains part of the public two-mode contract; this is its
        # exact-match fast path and intentionally bypasses vector inference.
        return _expand_explicit_for_missing_concepts(
            metadata, question, explicit
        ), "lexical"

    if db_identity:
        matches = vector_store.query_relevant_tables(
            question, db_identity=db_identity, top_k=top_k or MAX_RELEVANT_TABLES
        )
        if matches:
            lexical_signal = has_lexical_schema_signal(metadata, question)
            selected = {
                name
                for name, distance in matches
                if name in tables
                and (lexical_signal or float(distance) <= MAX_VECTOR_DISTANCE)
            }
            if lexical_signal:
                selected.update(
                    select_relevant_tables(metadata, question, hinted_tables)
                )
            names_by_lower = {name.casefold(): name for name in tables}
            for hint in hinted_tables or []:
                canonical = names_by_lower.get(hint.casefold())
                if canonical:
                    selected.add(canonical)
            if selected:
                selected |= _related_tables(metadata, selected)
                rank = {name: i for i, (name, _distance) in enumerate(matches)}
                ranked_names = sorted(selected, key=lambda n: rank.get(n, len(matches)))
                return ranked_names, "vector"

    return select_relevant_tables(metadata, question, hinted_tables), "lexical"


def _trim_table_columns(table: dict[str, Any], question: str) -> dict[str, Any]:
    columns = table.get("columns", {})
    if len(columns) <= MAX_COLUMNS_PER_TABLE:
        return table
    question_tokens = _tokenize(question)
    primary = set(table.get("primary_key", []))

    def priority(item: tuple[str, dict[str, Any]]) -> tuple[int, int, str]:
        name, column = item
        relevance = 5 * len(_tokenize(name) & question_tokens)
        relevance += 2 * len(
            _tokenize(column.get("description", "")) & question_tokens
        )
        structural = int(
            name in primary
            or column.get("is_foreign_key")
            or column.get("semantic_role") in {"measure", "temporal"}
        )
        return relevance, structural, name.casefold()

    ranked = sorted(columns.items(), key=priority, reverse=True)
    retained = {name for name, _column in ranked[:MAX_COLUMNS_PER_TABLE]} | primary
    trimmed = dict(table)
    trimmed["columns"] = {
        name: column for name, column in columns.items() if name in retained
    }
    trimmed["columns_omitted_from_prompt"] = len(columns) - len(trimmed["columns"])
    return trimmed


def _trim_metadata(
    metadata: dict[str, Any],
    relevant_names: list[str] | set[str],
    question: str = "",
) -> dict[str, Any]:
    relevant_names = set(relevant_names)
    tables = {
        name: _trim_table_columns(metadata["tables"][name], question)
        for name in relevant_names
    }
    relationships = [
        rel
        for rel in metadata.get("relationships", [])
        if rel["from_table"] in relevant_names
        and rel["to_table"] in relevant_names
        and rel["from_column"] in tables[rel["from_table"]].get("columns", {})
        and rel["to_column"] in tables[rel["to_table"]].get("columns", {})
    ]
    aggregation_rules = {
        name: {
            **rules,
            "measures": {
                column: aggregation
                for column, aggregation in rules.get("measures", {}).items()
                if column in tables[name].get("columns", {})
            },
            "default_measure": (
                rules.get("default_measure")
                if rules.get("default_measure") in tables[name].get("columns", {})
                else None
            ),
        }
        for name, rules in metadata.get("aggregation_rules", {}).items()
        if name in relevant_names
    }

    return {
        "tables": tables,
        "relationships": relationships,
        "aggregation_rules": aggregation_rules,
        "glossary": metadata.get("glossary", {}),
    }


def get_relevant_metadata(
    metadata: dict[str, Any],
    question: str,
    hinted_tables: list[str] | None = None,
    *,
    db_identity: str | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Return a trimmed metadata dict containing only the relevant tables.

    Without ``db_identity`` this is pure lexical scoring (unchanged
    behavior). Passing ``db_identity`` tries vector/RAG retrieval first; see
    :func:`get_relevant_metadata_with_mode` if the caller also wants to know
    which strategy was used.
    """
    relevant_names, _mode = select_relevant_tables_rag(
        metadata, question, hinted_tables, db_identity=db_identity, top_k=top_k
    )
    return _trim_metadata(metadata, relevant_names, question)


def get_relevant_metadata_with_mode(
    metadata: dict[str, Any],
    question: str,
    hinted_tables: list[str] | None = None,
    *,
    db_identity: str | None = None,
    top_k: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Like :func:`get_relevant_metadata`, plus which strategy answered ("vector"/"lexical")."""
    relevant_names, mode = select_relevant_tables_rag(
        metadata, question, hinted_tables, db_identity=db_identity, top_k=top_k
    )
    return _trim_metadata(metadata, relevant_names, question), mode


def format_metadata_for_prompt(relevant_metadata: dict[str, Any]) -> str:
    """Render trimmed metadata as compact, LLM-friendly text (not raw JSON)."""
    lines: list[str] = []

    for table_name, table in sorted(relevant_metadata.get("tables", {}).items()):
        pk = ", ".join(table.get("primary_key", [])) or "none"
        row_count_prefix = ">=" if table.get("row_count_is_lower_bound") else "~"
        lines.append(
            f"TABLE {table_name} ({table.get('object_type', 'table')}, "
            f"{table['kind']}, {row_count_prefix}{table['row_count']} rows, "
            f"primary key: {pk})"
        )
        if table.get("depends_on"):
            lines.append(
                "  derived from: " + ", ".join(table["depends_on"])
            )
        if table.get("columns_omitted_from_prompt"):
            lines.append(
                f"  note: {table['columns_omitted_from_prompt']} low-relevance columns "
                "were omitted from this bounded request context"
            )
        lines.append(f"  description: {table['description']}")
        for col_name, col in table.get("columns", {}).items():
            bits = [col["sql_type"], col["semantic_role"]]
            if col.get("is_foreign_key") and col.get("references"):
                ref = col["references"]
                source = col.get("relationship_source") or "declared"
                bits.append(f"{source} FK -> {ref['table']}.{ref['column']}")
            if col.get("default_aggregation"):
                bits.append(f"default_agg={col['default_aggregation']}")
            if col.get("sample_values"):
                sample = ", ".join(str(v) for v in col["sample_values"][:3])
                bits.append(f"e.g. [{sample}]")
            observed = col.get("observed_storage_types") or []
            if observed:
                bits.append(f"observed_types={','.join(observed)}")
            if col.get("observed_value_family") not in {None, "empty"}:
                bits.append(f"observed_family={col['observed_value_family']}")
            if col.get("sampled_null_fraction") is not None:
                bits.append(
                    f"sample_nulls={float(col['sampled_null_fraction']):.1%}"
                )
            lines.append(f"  - {col_name} ({', '.join(bits)}): {col['description']}")
        lines.append("")

    if relevant_metadata.get("relationships"):
        lines.append("RELATIONSHIPS:")
        for rel in relevant_metadata["relationships"]:
            source = rel.get("source", "declared")
            confidence = rel.get("confidence")
            confidence_text = (
                f", confidence={float(confidence):.0%}"
                if confidence is not None
                else ""
            )
            composite_text = (
                f", composite key part {int(rel.get('constraint_sequence', 0)) + 1}"
                f"/{int(rel['constraint_size'])}"
                if int(rel.get("constraint_size") or 1) > 1
                else ""
            )
            lines.append(
                f"  {rel['from_table']}.{rel['from_column']} -> "
                f"{rel['to_table']}.{rel['to_column']} "
                f"({source}{confidence_text}{composite_text})"
            )
        lines.append("")

    if relevant_metadata.get("aggregation_rules"):
        lines.append("AGGREGATION RULES:")
        for table_name, rules in relevant_metadata["aggregation_rules"].items():
            measures = ", ".join(f"{c}:{a}" for c, a in rules["measures"].items())
            default = rules.get("default_measure")
            default_txt = f" (default measure: {default})" if default else ""
            lines.append(f"  {table_name} measures{default_txt}: {measures}")
        lines.append("")

    if relevant_metadata.get("glossary"):
        lines.append("BUSINESS GLOSSARY:")
        for term, definition in relevant_metadata["glossary"].items():
            lines.append(f"  {term}: {definition}")

    return "\n".join(lines)
