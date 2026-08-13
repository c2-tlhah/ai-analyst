"""Versioned schema documentation with optional local vector retrieval.

Rather than only scoring tables by keyword overlap (see
:mod:`app.metadata.retrieval`'s lexical path), each table's business-enriched
metadata -- the same description/columns/sample-values content the LLM
enrichment step in :mod:`app.metadata.store` produces -- is rendered to a
short text document and always saved as a readable local knowledge base. When
enabled, the same content is embedded into an on-disk Chroma collection. A
question can then use nearest-neighbor search, which generalizes to paraphrased/semantic questions
that share no literal keywords with the schema (e.g. "top sellers" matching
a table described as "product sales facts") -- something the keyword scorer
alone cannot do.

Embeddings run entirely on-machine via Chroma's bundled ONNX MiniLM model:
no API key, no network call per query, and no LLM token cost.

**Versioning.** Every database gets its own numbered sequence of knowledge-
base versions, one per *actual* schema change (tracked via the structural
``schema_hash`` already computed by :mod:`app.metadata.discovery`) -- not one
per refresh. Re-syncing an unchanged schema (e.g. clicking "Refresh schema"
with nothing new) updates the current version's documents in place; a real
structural change (a column/table added, changed, or removed) mints a new
version and keeps every prior version's Chroma collection and on-disk text
export untouched, so nothing is ever silently overwritten or lost. Retrieval
(:func:`query_relevant_tables`) always searches the latest version.

**Documents are primary.** Every version's per-table documents are written as
plain ``.txt`` files under
``<vector store dir>/knowledge_base_txt/<db identity>/v<N>/`` (one file per
table, plus a combined ``_all_tables.txt``) so a user can inspect -- or diff
across versions -- exactly what the knowledge base knows. These files also
power lexical Q&A when Chroma is disabled or unhealthy.

Every function here degrades to returning ``None``/empty on any backend
failure (collection not yet built, corrupt store, import error) rather than
raising -- retrieval always has the lexical scorer to fall back to, and a
vector-store hiccup must never break question-answering.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

_client: Any = None  # lazy singleton; constructing a PersistentClient is not free
_last_errors: dict[str, str] = {}


@dataclass(frozen=True)
class RetrievedDocument:
    """One knowledge document returned by semantic similarity search."""

    table_name: str
    content: str
    distance: float
    version: int


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_client() -> Any:
    global _client
    if _client is None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        from app.config import get_settings

        settings = get_settings()
        settings.vector.directory.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(settings.vector.directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _client


def _vector_directory() -> Path:
    from app.config import get_settings

    return get_settings().vector.directory


def _collection_name(db_identity: str, version: int) -> str:
    # Chroma collection names must be 3-63 chars, alnum/underscore/hyphen.
    return f"schema_{db_identity}_v{version}"


def _table_document(
    table_name: str, table: dict[str, Any], metadata: dict[str, Any]
) -> str:
    """Render one table's full business-enriched metadata as embeddable text."""
    pk = ", ".join(table.get("primary_key", [])) or "none"
    count_prefix = ">=" if table.get("row_count_is_lower_bound") else "~"
    lines = [
        f"Table {table_name} ({table.get('object_type', 'table')}, "
        f"{table.get('kind', 'unknown')}, {count_prefix}{table.get('row_count', 0)} "
        f"rows, primary key: {pk})",
        table.get("description") or "",
    ]
    if table.get("depends_on"):
        lines.append("Derived from: " + ", ".join(table["depends_on"]))
    for col_name, col in table.get("columns", {}).items():
        bits = [b for b in (col.get("sql_type"), col.get("semantic_role")) if b]
        if col.get("is_foreign_key") and col.get("references"):
            ref = col["references"]
            source = col.get("relationship_source") or "declared"
            bits.append(f"{source} reference to {ref['table']}.{ref['column']}")
        if col.get("sample_values"):
            bits.append("examples: " + ", ".join(str(v) for v in col["sample_values"][:6]))
        if col.get("observed_value_family") not in {None, "empty"}:
            bits.append(f"observed family: {col['observed_value_family']}")
        if col.get("sampled_null_fraction") is not None:
            bits.append(f"sample nulls: {float(col['sampled_null_fraction']):.1%}")
        descriptor = f" ({', '.join(bits)})" if bits else ""
        lines.append(f"{col_name}{descriptor}: {col.get('description') or ''}")

    relationships = [
        rel
        for rel in metadata.get("relationships", [])
        if table_name in {rel.get("from_table"), rel.get("to_table")}
    ]
    if relationships:
        lines.append("")
        lines.append("Relationships:")
        for rel in relationships:
            source = rel.get("source", "declared")
            confidence = rel.get("confidence")
            confidence_text = (
                f", confidence {float(confidence):.0%}"
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
                f"- {rel['from_table']}.{rel['from_column']} -> "
                f"{rel['to_table']}.{rel['to_column']} "
                f"({source}{confidence_text}{composite_text})"
            )

    rules = metadata.get("aggregation_rules", {}).get(table_name, {})
    measures = rules.get("measures", {})
    if measures:
        default_measure = rules.get("default_measure") or "none"
        lines.append("")
        lines.append(f"Aggregation guidance (default measure: {default_measure}):")
        for column, aggregation in measures.items():
            lines.append(f"- {column}: {aggregation}")

    # Include only glossary definitions which explicitly refer to this table
    # or one of its columns. This enriches semantic retrieval without copying
    # the entire glossary into every document and making all vectors alike.
    identifiers = [table_name, *table.get("columns", {}).keys()]
    relevant_glossary = {
        term: definition
        for term, definition in metadata.get("glossary", {}).items()
        if any(identifier.casefold() in str(definition).casefold() for identifier in identifiers)
    }
    if relevant_glossary:
        lines.append("")
        lines.append("Relevant business glossary:")
        for term, definition in relevant_glossary.items():
            lines.append(f"- {term}: {definition}")
    return "\n".join(lines)


def render_documents(metadata: dict[str, Any]) -> dict[str, str]:
    """Render the complete, human-readable document set for ``metadata``.

    Document generation is intentionally independent from vector indexing.
    A usable knowledge base therefore still exists when Chroma is disabled or
    temporarily unavailable; semantic indexing is only an optional accelerator.
    """
    tables = metadata.get("tables", {})
    return {
        name: _table_document(name, tables[name], metadata)
        for name in sorted(tables)
    }


def _clean_error(exc: BaseException) -> str:
    detail = " ".join(str(exc).split()) or "No detail was supplied by the vector backend."
    return f"{type(exc).__name__}: {detail}"[:500]


# ---------------------------------------------------------------------------
# Version registry: one small JSON file per database, listing every version
# ever built (never rewritten in place -- only appended to). This is the
# source of truth for "which collection is current" and "what versions
# exist"; Chroma collections and the .txt export are derived from it.
# ---------------------------------------------------------------------------


def _versions_file(db_identity: str) -> Path:
    directory = _vector_directory() / "versions"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{db_identity}.json"


def _load_versions(db_identity: str) -> list[dict[str, Any]]:
    path = _versions_file(db_identity)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Could not read knowledge-base version registry for db %s", db_identity)
        return []


def _save_versions(db_identity: str, versions: list[dict[str, Any]]) -> None:
    _versions_file(db_identity).write_text(json.dumps(versions, indent=2), encoding="utf-8")


def list_versions(db_identity: str) -> list[dict[str, Any]]:
    """Every knowledge-base version built for this database, oldest first."""
    return _load_versions(db_identity)


def _latest_version(db_identity: str) -> dict[str, Any] | None:
    versions = _load_versions(db_identity)
    return versions[-1] if versions else None


# ---------------------------------------------------------------------------
# Text export
# ---------------------------------------------------------------------------


def _safe_filename(table_name: str) -> str:
    return table_name.replace("/", "_").replace("\\", "_")


def _txt_dir(db_identity: str, version: int) -> Path:
    return _vector_directory() / "knowledge_base_txt" / db_identity / f"v{version}"


def knowledge_document_files(
    db_identity: str, version: int, documents: dict[str, str]
) -> dict[Path, str]:
    """Return the exact managed file set for a knowledge-base version."""
    directory = _txt_dir(db_identity, version)
    files = {
        directory / f"{_safe_filename(table_name)}.txt": text
        for table_name, text in sorted(documents.items())
    }
    combined = [
        f"{'=' * 70}\n{table_name}\n{'=' * 70}\n{text}\n"
        for table_name, text in sorted(documents.items())
    ]
    files[directory / "_all_tables.txt"] = "\n".join(combined)
    return files


def _write_txt_documents(db_identity: str, version: int, documents: dict[str, str]) -> None:
    directory = _txt_dir(db_identity, version)
    directory.mkdir(parents=True, exist_ok=True)
    for path, text in knowledge_document_files(db_identity, version, documents).items():
        path.write_text(text, encoding="utf-8")


def read_version_documents(db_identity: str, version: int) -> dict[str, str]:
    """Read the human-readable per-table documents saved for one version."""
    directory = _txt_dir(db_identity, version)
    if not directory.exists():
        return {}
    docs: dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        if path.stem == "_all_tables":
            continue
        docs[path.stem] = path.read_text(encoding="utf-8")
    return docs


# ---------------------------------------------------------------------------
# Sync / query
# ---------------------------------------------------------------------------


def sync_collection(
    metadata: dict[str, Any],
    *,
    db_identity: str,
    documents: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Save knowledge documents and optionally synchronize their vector index.

    Text documents are the source of truth and are always versioned/written.
    Chroma is an optional semantic-search layer. If it is disabled, callers can
    immediately use lexical document search; if it fails, the failure is
    reported while the generated documentation remains readable.
    """
    from app.config import get_settings

    tables = metadata.get("tables", {})
    if not tables:
        _last_errors[db_identity] = "The schema contains no tables to index."
        return None

    schema_hash = metadata.get("schema_hash", "")
    documents = documents or render_documents(metadata)
    metadatas = [{"table": name, "kind": tables[name].get("kind", "unknown")} for name in documents]

    latest = _latest_version(db_identity)
    reuse_current_version = latest is not None and latest.get("schema_hash") == schema_hash

    if reuse_current_version:
        version_record = latest
    else:
        versions = _load_versions(db_identity)
        next_version = (latest["version"] + 1) if latest else 1
        collection_name = _collection_name(db_identity, next_version)
        version_record = {
            "version": next_version,
            "schema_hash": schema_hash,
            "created_at": _now_iso(),
            "tables": sorted(documents),
            "collection": collection_name,
        }
        versions.append(version_record)
        _save_versions(db_identity, versions)

    # Keep the plain-text knowledge base available regardless of vector
    # configuration/backend health.
    _write_txt_documents(db_identity, version_record["version"], documents)

    if not get_settings().vector.enabled:
        _last_errors.pop(db_identity, None)
        logger.info(
            "Knowledge documents saved: %d table(s), db %s, version %d (vector disabled)",
            len(documents),
            db_identity,
            version_record["version"],
        )
        return version_record

    try:
        client = _get_client()
        if reuse_current_version:
            collection = client.get_or_create_collection(latest["collection"])
            collection.upsert(ids=list(documents), documents=list(documents.values()), metadatas=metadatas)
        else:
            collection = client.get_or_create_collection(version_record["collection"])
            collection.upsert(ids=list(documents), documents=list(documents.values()), metadatas=metadatas)
    except Exception as exc:  # noqa: BLE001 - indexing must never break a schema refresh
        _last_errors[db_identity] = _clean_error(exc)
        logger.exception("Vector store sync failed for db %s", db_identity)
        return None

    _last_errors.pop(db_identity, None)
    logger.info(
        "Knowledge base synced: %d table(s), db %s, version %d%s",
        len(documents),
        db_identity,
        version_record["version"],
        " (new version)" if not reuse_current_version else "",
    )
    return version_record


def _document_tokens(text: str) -> set[str]:
    import re

    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return {
        token.casefold()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_]+", text)
        if len(token) > 1
    }


def query_documents_with_fallback(
    question: str,
    *,
    db_identity: str,
    top_k: int,
    fallback_documents: dict[str, str] | None = None,
) -> tuple[list[RetrievedDocument], str]:
    """Search documents semantically, then fall back to local lexical ranking."""
    semantic = query_relevant_documents(
        question,
        db_identity=db_identity,
        top_k=top_k,
    )
    if semantic:
        return semantic, "vector"

    latest = _latest_version(db_identity)
    documents = (
        read_version_documents(db_identity, int(latest["version"]))
        if latest is not None
        else dict(fallback_documents or {})
    )
    if not documents and fallback_documents:
        documents = dict(fallback_documents)
    if not documents:
        return [], "lexical"

    query_tokens = _document_tokens(question)
    ranked: list[tuple[int, str, str]] = []
    for table_name, content in documents.items():
        table_tokens = _document_tokens(table_name)
        content_tokens = _document_tokens(content)
        score = 4 * len(query_tokens & table_tokens) + len(query_tokens & content_tokens)
        ranked.append((score, table_name, content))
    ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
    selected = ranked[: min(max(1, top_k), len(ranked))]
    return (
        [
            RetrievedDocument(
                table_name=table_name,
                content=content,
                # Distance is informational in the UI. Lower remains better.
                distance=1.0 / (1.0 + max(0, score)),
                version=int(latest["version"]) if latest is not None else 0,
            )
            for score, table_name, content in selected
        ],
        "lexical",
    )


def query_relevant_documents(
    question: str, *, db_identity: str, top_k: int
) -> list[RetrievedDocument] | None:
    """Return the latest semantically relevant knowledge documents.

    Returns ``None`` (never raises) when nothing is indexed yet for this
    database or the query otherwise fails -- callers should fall back to
    another retrieval strategy or show an actionable build error.
    """
    from app.config import get_settings

    if not get_settings().vector.enabled:
        return None

    latest = _latest_version(db_identity)
    if latest is None:
        return None

    try:
        collection = _get_client().get_collection(latest["collection"])
        count = collection.count()
        if count == 0:
            return None
        result = collection.query(
            query_texts=[question],
            n_results=min(max(1, top_k), count),
            include=["documents", "distances", "metadatas"],
        )
    except Exception as exc:  # noqa: BLE001 - collection missing, backend error, etc.
        _last_errors[db_identity] = _clean_error(exc)
        return None

    ids = (result.get("ids") or [[]])[0]
    contents = (result.get("documents") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    hits: list[RetrievedDocument] = []
    for index, table_name in enumerate(ids):
        if index >= len(contents) or not contents[index]:
            continue
        distance = float(distances[index]) if index < len(distances) else 0.0
        hits.append(
            RetrievedDocument(
                table_name=str(table_name),
                content=str(contents[index]),
                distance=distance,
                version=int(latest["version"]),
            )
        )
    return hits or None


def query_relevant_tables(
    question: str, *, db_identity: str, top_k: int
) -> list[tuple[str, float]] | None:
    """Return table names/distances for the SQL schema-retrieval pipeline."""
    documents = query_relevant_documents(
        question,
        db_identity=db_identity,
        top_k=top_k,
    )
    if not documents:
        return None
    return [(document.table_name, document.distance) for document in documents]


def collection_stats(db_identity: str) -> dict[str, Any]:
    """Return an actionable knowledge-base health summary for UI/backend use."""
    from app.config import get_settings

    settings = get_settings()
    versions = _load_versions(db_identity)
    base = {
        "enabled": settings.vector.enabled,
        "indexed": False,
        "table_count": 0,
        "document_count": 0,
        "version": 0,
        "version_count": len(versions),
        "status": "disabled" if not settings.vector.enabled else "not_built",
        "error": None,
        "text_export_path": None,
    }
    if not versions:
        if db_identity in _last_errors:
            base.update(status="error", error=_last_errors[db_identity])
        return base

    latest = versions[-1]
    version = latest["version"]
    text_documents = read_version_documents(db_identity, version)
    base.update(
        table_count=len(latest.get("tables", [])),
        document_count=len(text_documents),
        version=version,
        created_at=latest.get("created_at"),
        text_export_path=str(_txt_dir(db_identity, version)),
    )
    if not settings.vector.enabled:
        base.update(status="documents_ready" if text_documents else "not_built")
        return base
    if db_identity in _last_errors:
        base.update(status="error", error=_last_errors[db_identity])
        return base
    try:
        document_count = _get_client().get_collection(latest["collection"]).count()
        if document_count <= 0:
            raise RuntimeError("The latest vector collection is empty.")
    except Exception as exc:  # noqa: BLE001 - health checks must be safe for the UI
        error = _last_errors.get(db_identity) or _clean_error(exc)
        _last_errors[db_identity] = error
        base.update(status="error", error=error)
        return base

    _last_errors.pop(db_identity, None)
    base.update(indexed=True, document_count=document_count, status="ready")
    return base
