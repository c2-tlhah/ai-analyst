"""Local, versioned vector knowledge base for schema metadata (RAG retrieval).

Rather than only scoring tables by keyword overlap (see
:mod:`app.metadata.retrieval`'s lexical path), each table's business-enriched
metadata -- the same description/columns/sample-values content the LLM
enrichment step in :mod:`app.metadata.store` produces -- is rendered to a
short text document and embedded into a local, on-disk Chroma collection. A
question is answered by embedding it and doing a nearest-neighbor search
over those documents, which generalizes to paraphrased/semantic questions
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

**Text export.** Alongside each version's Chroma collection, the exact same
per-table documents are written as plain ``.txt`` files under
``<vector store dir>/knowledge_base_txt/<db identity>/v<N>/`` (one file per
table, plus a combined ``_all_tables.txt``) so a user can inspect -- or diff
across versions -- exactly what the knowledge base knows, without a Chroma
client.

Every function here degrades to returning ``None``/empty on any backend
failure (collection not yet built, corrupt store, import error) rather than
raising -- retrieval always has the lexical scorer to fall back to, and a
vector-store hiccup must never break question-answering.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging_config import get_logger

logger = get_logger(__name__)

_client: Any = None  # lazy singleton; constructing a PersistentClient is not free


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


def _table_document(table_name: str, table: dict[str, Any]) -> str:
    """Render one table's full business-enriched metadata as embeddable text."""
    pk = ", ".join(table.get("primary_key", [])) or "none"
    lines = [
        f"Table {table_name} ({table.get('kind', 'unknown')}, "
        f"~{table.get('row_count', 0)} rows, primary key: {pk})",
        table.get("description") or "",
    ]
    for col_name, col in table.get("columns", {}).items():
        bits = [b for b in (col.get("sql_type"), col.get("semantic_role")) if b]
        if col.get("is_foreign_key") and col.get("references"):
            ref = col["references"]
            bits.append(f"references {ref['table']}.{ref['column']}")
        if col.get("sample_values"):
            bits.append("examples: " + ", ".join(str(v) for v in col["sample_values"][:6]))
        descriptor = f" ({', '.join(bits)})" if bits else ""
        lines.append(f"{col_name}{descriptor}: {col.get('description') or ''}")
    return "\n".join(lines)


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


def _write_txt_documents(db_identity: str, version: int, documents: dict[str, str]) -> None:
    directory = _txt_dir(db_identity, version)
    directory.mkdir(parents=True, exist_ok=True)
    combined: list[str] = []
    for table_name, text in sorted(documents.items()):
        (directory / f"{_safe_filename(table_name)}.txt").write_text(text, encoding="utf-8")
        combined.append(f"{'=' * 70}\n{table_name}\n{'=' * 70}\n{text}\n")
    (directory / "_all_tables.txt").write_text("\n".join(combined), encoding="utf-8")


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


def sync_collection(metadata: dict[str, Any], *, db_identity: str) -> dict[str, Any] | None:
    """Bring the vector knowledge base for one database up to date.

    Mints a new version when the schema's structural hash differs from the
    latest known version (or none exists yet); otherwise refreshes the
    current version's documents in place (e.g. an unchanged-schema "Refresh
    schema" click just re-embeds, it doesn't fork a new version). Either way
    the same documents are written to the on-disk ``.txt`` export.

    Returns the version record that's now current, or ``None`` if there was
    nothing to index or the vector backend failed (logged, never raised).
    """
    tables = metadata.get("tables", {})
    if not tables:
        return None

    schema_hash = metadata.get("schema_hash", "")
    documents = {name: _table_document(name, tables[name]) for name in tables}
    metadatas = [{"table": name, "kind": tables[name].get("kind", "unknown")} for name in documents]

    latest = _latest_version(db_identity)
    reuse_current_version = latest is not None and latest.get("schema_hash") == schema_hash

    try:
        client = _get_client()
        if reuse_current_version:
            collection = client.get_or_create_collection(latest["collection"])
            collection.upsert(ids=list(documents), documents=list(documents.values()), metadatas=metadatas)
            version_record = latest
        else:
            versions = _load_versions(db_identity)
            next_version = (latest["version"] + 1) if latest else 1
            collection_name = _collection_name(db_identity, next_version)
            collection = client.get_or_create_collection(collection_name)
            collection.upsert(ids=list(documents), documents=list(documents.values()), metadatas=metadatas)
            version_record = {
                "version": next_version,
                "schema_hash": schema_hash,
                "created_at": _now_iso(),
                "tables": sorted(documents),
                "collection": collection_name,
            }
            versions.append(version_record)
            _save_versions(db_identity, versions)
    except Exception:  # noqa: BLE001 - indexing must never break a schema refresh
        logger.exception("Vector store sync failed for db %s", db_identity)
        return None

    _write_txt_documents(db_identity, version_record["version"], documents)
    logger.info(
        "Knowledge base synced: %d table(s), db %s, version %d%s",
        len(documents),
        db_identity,
        version_record["version"],
        " (new version)" if not reuse_current_version else "",
    )
    return version_record


def query_relevant_tables(
    question: str, *, db_identity: str, top_k: int
) -> list[tuple[str, float]] | None:
    """Return ``[(table_name, distance), ...]`` from the latest version, ranked
    most-similar first.

    Returns ``None`` (never raises) when nothing is indexed yet for this
    database or the query otherwise fails -- callers should fall back to
    lexical retrieval in that case.
    """
    latest = _latest_version(db_identity)
    if latest is None:
        return None

    try:
        collection = _get_client().get_collection(latest["collection"])
        count = collection.count()
        if count == 0:
            return None
        result = collection.query(query_texts=[question], n_results=min(top_k, count))
    except Exception:  # noqa: BLE001 - collection missing, backend error, etc.
        return None

    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return list(zip(ids, distances))


def collection_stats(db_identity: str) -> dict[str, Any]:
    """Latest-version summary for a database, for UI display only."""
    versions = _load_versions(db_identity)
    if not versions:
        return {"indexed": False, "table_count": 0, "version": 0, "version_count": 0}
    latest = versions[-1]
    return {
        "indexed": True,
        "table_count": len(latest.get("tables", [])),
        "version": latest["version"],
        "version_count": len(versions),
        "created_at": latest.get("created_at"),
    }
