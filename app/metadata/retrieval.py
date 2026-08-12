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

_COLUMN_NAME_WEIGHT = 3
_TABLE_NAME_WEIGHT = 4
_DESCRIPTION_WEIGHT = 1
_SAMPLE_VALUE_WEIGHT = 2


def _tokenize(text: str) -> set[str]:
    # Natural-language "sales amount" should match identifiers such as
    # SalesAmount; split lower-to-upper camel-case boundaries before tokenizing.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return {
        w.lower()
        for w in _TOKEN_RE.findall(text)
        if len(w) > 1 and w.lower() not in _STOPWORDS
    }


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
            if str(sample).lower() in question_tokens:
                score += _SAMPLE_VALUE_WEIGHT

    return score


def _related_tables(metadata: dict[str, Any], table_names: set[str]) -> set[str]:
    related = set()
    for rel in metadata.get("relationships", []):
        if rel["from_table"] in table_names:
            related.add(rel["to_table"])
        if rel["to_table"] in table_names:
            related.add(rel["from_table"])
    return related


def select_relevant_tables(
    metadata: dict[str, Any],
    question: str,
    hinted_tables: list[str] | None = None,
) -> list[str]:
    """Pick the tables relevant to ``question``, expanded with FK neighbors."""
    tables = metadata.get("tables", {})
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

    if db_identity:
        matches = vector_store.query_relevant_tables(
            question, db_identity=db_identity, top_k=top_k or MAX_RELEVANT_TABLES
        )
        if matches:
            selected = {name for name, _distance in matches if name in tables}
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


def _trim_metadata(metadata: dict[str, Any], relevant_names: list[str] | set[str]) -> dict[str, Any]:
    relevant_names = set(relevant_names)
    tables = {name: metadata["tables"][name] for name in relevant_names}
    relationships = [
        rel
        for rel in metadata.get("relationships", [])
        if rel["from_table"] in relevant_names and rel["to_table"] in relevant_names
    ]
    aggregation_rules = {
        name: rules
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
    return _trim_metadata(metadata, relevant_names)


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
    return _trim_metadata(metadata, relevant_names), mode


def format_metadata_for_prompt(relevant_metadata: dict[str, Any]) -> str:
    """Render trimmed metadata as compact, LLM-friendly text (not raw JSON)."""
    lines: list[str] = []

    for table_name, table in sorted(relevant_metadata.get("tables", {}).items()):
        pk = ", ".join(table.get("primary_key", [])) or "none"
        lines.append(
            f"TABLE {table_name} ({table['kind']}, ~{table['row_count']} rows, "
            f"primary key: {pk})"
        )
        lines.append(f"  description: {table['description']}")
        for col_name, col in table.get("columns", {}).items():
            bits = [col["sql_type"], col["semantic_role"]]
            if col.get("is_foreign_key") and col.get("references"):
                ref = col["references"]
                bits.append(f"FK -> {ref['table']}.{ref['column']}")
            if col.get("default_aggregation"):
                bits.append(f"default_agg={col['default_aggregation']}")
            if col.get("sample_values"):
                sample = ", ".join(str(v) for v in col["sample_values"][:8])
                bits.append(f"e.g. [{sample}]")
            lines.append(f"  - {col_name} ({', '.join(bits)}): {col['description']}")
        lines.append("")

    if relevant_metadata.get("relationships"):
        lines.append("RELATIONSHIPS:")
        for rel in relevant_metadata["relationships"]:
            lines.append(
                f"  {rel['from_table']}.{rel['from_column']} -> "
                f"{rel['to_table']}.{rel['to_column']}"
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
