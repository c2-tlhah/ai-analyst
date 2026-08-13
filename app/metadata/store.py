"""Persistent metadata store.

Combines deterministic schema :mod:`~app.metadata.discovery` with a
business-context layer (descriptions, glossary, aggregation overrides) and
persists the merged result as database-identity-scoped JSON under
``metadata_store/``. This is the
context layer the rest of the backend reads from instead of re-inspecting
the database (or sending the raw schema) on every request.

Two files are maintained per connected database:

* ``business_context.json`` -- human/LLM-authored descriptions, keyed by
  table/column name. Survives schema rebuilds; new tables/columns are
  appended to it (never silently dropped), so curation accumulates.
* ``schema_metadata.json`` -- the full merged metadata consumed by the
  application, plus a structural hash used to detect drift.

Call :func:`refresh_if_needed` on startup and periodic cache verification: it
compares the live schema plus database/WAL revision and only re-persists when
the schema, profiled data revision, or metadata contract changed.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from inspect import Parameter, signature
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import get_settings
from app.db.connection import get_active_database_identity, get_active_database_path
from app.logging_config import get_logger
from app.metadata import discovery

logger = get_logger(__name__)
METADATA_FORMAT_VERSION = 5

# enrich_fn(table_name, kind, column_names, schema_context)
#   -> {"table": str, "columns": {col: str}}
EnrichFn = Callable[[str, str, list[str], dict[str, Any]], dict[str, Any]]

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_fingerprint() -> dict[str, int]:
    """Cheap data-change signal, including SQLite WAL-backed updates."""
    path = get_active_database_path()
    result: dict[str, int] = {}
    for label, candidate in (("database", path), ("wal", Path(f"{path}-wal"))):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        result[f"{label}_size"] = int(stat.st_size)
        result[f"{label}_mtime_ns"] = int(stat.st_mtime_ns)
    return result


def metadata_paths() -> tuple[Path, Path]:
    """Return schema/context files isolated for the active database.

    The configured default database keeps the legacy top-level paths so
    existing curated installations migrate without losing their context.
    Every other database receives a stable identity-scoped directory.
    """
    settings = get_settings()
    active = get_active_database_path().resolve(strict=False)
    configured = settings.database.path.resolve(strict=False)
    if active == configured:
        return settings.metadata.schema_file, settings.metadata.business_context_file
    identity = get_active_database_identity()
    return (
        settings.metadata.schema_file_for(identity),
        settings.metadata.business_context_file_for(identity),
    )


def load_business_context(path: Path | None = None) -> dict[str, Any]:
    p = path or metadata_paths()[1]
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"tables": {}, "glossary": {}}


def save_business_context(ctx: dict[str, Any], path: Path | None = None) -> None:
    p = path or metadata_paths()[1]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, sort_keys=True)


def _describe_table_and_columns(
    table: discovery.TableInfo,
    business_ctx: dict[str, Any],
    enrich_fn: Optional[EnrichFn],
    generated_override: dict[str, Any] | None = None,
) -> tuple[str, dict[str, str], dict[str, str], str | None]:
    """Return (table_description, {col: description}, {col: agg_override}, default_measure)."""
    tables_ctx = business_ctx.setdefault("tables", {})
    entry = tables_ctx.get(table.name)

    known_columns = set(entry["columns"].keys()) if entry else set()
    live_columns = {c.name for c in table.columns}
    missing_columns = live_columns - known_columns

    if entry is None or missing_columns:
        if generated_override is not None:
            generated = generated_override
        elif enrich_fn is not None:
            try:
                requested_columns = sorted(missing_columns) or sorted(live_columns)
                schema_context = {
                    "row_count": table.row_count,
                    "columns": [column.to_dict() for column in table.columns],
                }
                function_parameters = signature(enrich_fn).parameters
                supports_context = any(
                    parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}
                    for parameter in function_parameters.values()
                ) or len(function_parameters) >= 4
                generated = (
                    enrich_fn(table.name, table.kind, requested_columns, schema_context)
                    if supports_context
                    else enrich_fn(table.name, table.kind, requested_columns)  # type: ignore[call-arg]
                )
            except Exception:  # noqa: BLE001 - enrichment must never break discovery
                logger.exception("LLM metadata enrichment failed for table %s", table.name)
                generated = {}
        else:
            generated = {}

        if entry is None:
            entry = {
                "description": generated.get("table")
                or f"{discovery.humanize(table.name)} table.",
                "columns": {},
                "aggregation_overrides": {},
                "default_measure": None,
                "source": "llm" if generated else "heuristic",
            }
            tables_ctx[table.name] = entry

        gen_cols = generated.get("columns", {}) if generated else {}
        for col in missing_columns:
            entry["columns"][col] = gen_cols.get(col) or discovery.humanize(col)
            entry.setdefault("source", "heuristic")

    return (
        entry.get("description") or f"{discovery.humanize(table.name)} table.",
        dict(entry.get("columns", {})),
        dict(entry.get("aggregation_overrides", {})),
        entry.get("default_measure"),
    )


def build_metadata(
    conn: sqlite3.Connection,
    business_ctx: dict[str, Any] | None = None,
    enrich_fn: Optional[EnrichFn] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Discover the live schema and merge it with business context.

    Returns ``(metadata, updated_business_ctx)`` -- the caller is
    responsible for persisting both (see :func:`refresh_if_needed`).
    """
    tables = discovery.discover_schema(conn)
    business_ctx = business_ctx if business_ctx is not None else load_business_context()
    business_ctx.setdefault("tables", {})
    business_ctx.setdefault("glossary", {})
    schema_hash = discovery.schema_signature(tables)

    # New databases often contain many tables. Enrich all missing descriptions
    # in configurable batches rather than consuming one provider request per
    # table. Plain callable enrichers keep their original per-table behavior.
    batch_generated: dict[str, dict[str, Any]] | None = None
    enrich_many = getattr(enrich_fn, "enrich_many", None) if enrich_fn else None
    if callable(enrich_many):
        requests: list[dict[str, Any]] = []
        tables_ctx = business_ctx.setdefault("tables", {})
        for table in tables.values():
            entry = tables_ctx.get(table.name)
            known_columns = set(entry.get("columns", {})) if entry else set()
            live_columns = {column.name for column in table.columns}
            missing_columns = live_columns - known_columns
            if entry is None or missing_columns:
                requests.append(
                    {
                        "table_name": table.name,
                        "kind": table.kind,
                        "columns": sorted(missing_columns) or sorted(live_columns),
                        "schema_context": {
                            "row_count": table.row_count,
                            "columns": [column.to_dict() for column in table.columns],
                        },
                    }
                )
        try:
            batch_generated = enrich_many(requests) if requests else {}
        except Exception:  # noqa: BLE001 - heuristics remain available
            logger.exception("Batched metadata enrichment failed")
            batch_generated = {}

    metadata_tables: dict[str, Any] = {}
    relationships: list[dict[str, str]] = []
    aggregation_rules: dict[str, Any] = {}

    for name, table in sorted(tables.items()):
        table_desc, col_desc, agg_overrides, default_measure = _describe_table_and_columns(
            table,
            business_ctx,
            enrich_fn,
            (
                batch_generated.get(table.name, {})
                if batch_generated is not None
                else None
            ),
        )

        columns: dict[str, Any] = {}
        measures: dict[str, str] = {}
        primary_key: list[str] = []

        for col in table.columns:
            col_dict = col.to_dict()
            col_dict["description"] = col_desc.get(col.name, discovery.humanize(col.name))
            if col.name in agg_overrides:
                col_dict["default_aggregation"] = agg_overrides[col.name]
            columns[col.name] = col_dict

            if col.is_primary_key:
                primary_key.append(col.name)
            if col_dict["default_aggregation"]:
                measures[col.name] = col_dict["default_aggregation"]

            if col.references:
                relationships.append(
                    {
                        "from_table": table.name,
                        "from_column": col.name,
                        "to_table": col.references["table"],
                        "to_column": col.references["column"],
                        "source": col.relationship_source or "declared",
                        "confidence": 1.0 if col.relationship_source == "declared" else 0.8,
                        "constraint_id": col.relationship_constraint_id,
                        "constraint_sequence": col.relationship_constraint_sequence,
                        "constraint_size": col.relationship_constraint_size,
                    }
                )

        metadata_tables[name] = {
            "kind": table.kind,
            "object_type": table.object_type,
            "depends_on": table.depends_on,
            "description": table_desc,
            "row_count": table.row_count,
            "row_count_is_lower_bound": table.row_count_is_lower_bound,
            "primary_key": primary_key,
            "columns": columns,
        }

        if measures:
            aggregation_rules[name] = {
                "measures": measures,
                "default_measure": default_measure if default_measure in measures else None,
            }

    metadata = {
        "metadata_format_version": METADATA_FORMAT_VERSION,
        "source_fingerprint": _source_fingerprint(),
        "generated_at": _now_iso(),
        "database_identity": get_active_database_identity(),
        "database_path": str(get_active_database_path()),
        "schema_hash": schema_hash,
        "tables": metadata_tables,
        "relationships": relationships,
        "aggregation_rules": aggregation_rules,
        "glossary": business_ctx.get("glossary", {}),
    }
    return metadata, business_ctx


def load_schema_metadata(path: Path | None = None) -> dict[str, Any] | None:
    p = path or metadata_paths()[0]
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def cached_metadata_matches_active_source(metadata: dict[str, Any] | None) -> bool:
    """Cheap cache guard used by MCP calls before paying for rediscovery."""
    return bool(
        metadata
        and metadata.get("metadata_format_version") == METADATA_FORMAT_VERSION
        and metadata.get("database_identity") == get_active_database_identity()
        and metadata.get("source_fingerprint") == _source_fingerprint()
    )


def save_schema_metadata(metadata: dict[str, Any], path: Path | None = None) -> None:
    p = path or metadata_paths()[0]
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)


def refresh_if_needed(
    conn: sqlite3.Connection,
    enrich_fn: Optional[EnrichFn] = None,
    force: bool = False,
) -> tuple[dict[str, Any], bool]:
    """Load cached metadata, or rebuild it if the live schema drifted.

    Returns ``(metadata, was_rebuilt)``.
    """
    live_tables = discovery.discover_schema(conn)
    live_hash = discovery.schema_signature(live_tables)

    cached = load_schema_metadata()
    active_identity = get_active_database_identity()
    cached_identity = cached.get("database_identity") if cached else None
    # Legacy default metadata predates database_identity; the dynamically
    # selected path already proves it belongs to the configured default DB.
    identity_matches = cached_identity in {None, active_identity}
    if (
        not force
        and cached is not None
        and identity_matches
        and cached.get("metadata_format_version") == METADATA_FORMAT_VERSION
        and cached.get("schema_hash") == live_hash
        and cached.get("source_fingerprint") == _source_fingerprint()
    ):
        return cached, False

    logger.info(
        "Schema change detected (or no cache present); rebuilding metadata store."
    )
    metadata, business_ctx = build_metadata(conn, enrich_fn=enrich_fn)
    save_schema_metadata(metadata)
    save_business_context(business_ctx)
    return metadata, True


def get_table_catalog(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Lightweight table list (name/kind/description/row_count) -- no columns.

    Cheap enough to hand to the LLM for intent classification without
    paying for the full schema on every request.
    """
    return [
        {
            "name": name,
            "kind": info["kind"],
            "object_type": info.get("object_type", "table"),
            "description": info["description"],
            "row_count": info["row_count"],
            "row_count_is_lower_bound": info.get("row_count_is_lower_bound", False),
        }
        for name, info in sorted(metadata.get("tables", {}).items())
    ]
