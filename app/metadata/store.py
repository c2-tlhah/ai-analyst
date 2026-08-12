"""Persistent metadata store.

Combines deterministic schema :mod:`~app.metadata.discovery` with a
business-context layer (descriptions, glossary, aggregation overrides) and
persists the merged result as JSON under ``metadata_store/``. This is the
context layer the rest of the backend reads from instead of re-inspecting
the database (or sending the raw schema) on every request.

Two files are maintained:

* ``business_context.json`` -- human/LLM-authored descriptions, keyed by
  table/column name. Survives schema rebuilds; new tables/columns are
  appended to it (never silently dropped), so curation accumulates.
* ``schema_metadata.json`` -- the full merged metadata consumed by the
  application, plus a structural hash used to detect drift.

Call :func:`refresh_if_needed` on startup (and it is cheap enough to call
before every request): it hashes the live schema, and only re-discovers /
re-persists when something actually changed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from app.config import get_settings
from app.logging_config import get_logger
from app.metadata import discovery
from app.metadata.business_context_seed import SEED_BUSINESS_CONTEXT

logger = get_logger(__name__)

# enrich_fn(table_name, kind, column_names) -> {"table": str, "columns": {col: str}}
EnrichFn = Callable[[str, str, list[str]], dict[str, Any]]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_business_context(path: Path | None = None) -> dict[str, Any]:
    settings = get_settings()
    p = path or settings.metadata.business_context_file
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"tables": {}, "glossary": {}}


def save_business_context(ctx: dict[str, Any], path: Path | None = None) -> None:
    settings = get_settings()
    p = path or settings.metadata.business_context_file
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(ctx, f, indent=2, sort_keys=True)


def _ensure_seed(ctx: dict[str, Any]) -> dict[str, Any]:
    """Merge the curated seed into an existing business-context file.

    Curated entries win on first write; once a table exists in the on-disk
    context (curated or previously auto-generated) it is left alone here.
    """
    changed = False
    ctx.setdefault("tables", {})
    ctx.setdefault("glossary", {})
    for table, seed_table in SEED_BUSINESS_CONTEXT["tables"].items():
        if table not in ctx["tables"]:
            ctx["tables"][table] = {**seed_table, "source": "curated"}
            changed = True
    for term, definition in SEED_BUSINESS_CONTEXT["glossary"].items():
        if term not in ctx["glossary"]:
            ctx["glossary"][term] = definition
            changed = True
    return ctx if changed else ctx


def _describe_table_and_columns(
    table: discovery.TableInfo,
    business_ctx: dict[str, Any],
    enrich_fn: Optional[EnrichFn],
) -> tuple[str, dict[str, str], dict[str, str], str | None]:
    """Return (table_description, {col: description}, {col: agg_override}, default_measure)."""
    tables_ctx = business_ctx.setdefault("tables", {})
    entry = tables_ctx.get(table.name)

    known_columns = set(entry["columns"].keys()) if entry else set()
    live_columns = {c.name for c in table.columns}
    missing_columns = live_columns - known_columns

    if entry is None or missing_columns:
        if enrich_fn is not None:
            try:
                generated = enrich_fn(
                    table.name, table.kind, sorted(missing_columns) or sorted(live_columns)
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
    business_ctx = _ensure_seed(business_ctx if business_ctx is not None else load_business_context())

    tables = discovery.discover_schema(conn)
    schema_hash = discovery.schema_signature(tables)

    metadata_tables: dict[str, Any] = {}
    relationships: list[dict[str, str]] = []
    aggregation_rules: dict[str, Any] = {}

    for name, table in sorted(tables.items()):
        table_desc, col_desc, agg_overrides, default_measure = _describe_table_and_columns(
            table, business_ctx, enrich_fn
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
                    }
                )

        metadata_tables[name] = {
            "kind": table.kind,
            "description": table_desc,
            "row_count": table.row_count,
            "primary_key": primary_key,
            "columns": columns,
        }

        if measures:
            aggregation_rules[name] = {
                "measures": measures,
                "default_measure": default_measure if default_measure in measures else None,
            }

    metadata = {
        "generated_at": _now_iso(),
        "schema_hash": schema_hash,
        "tables": metadata_tables,
        "relationships": relationships,
        "aggregation_rules": aggregation_rules,
        "glossary": business_ctx.get("glossary", {}),
    }
    return metadata, business_ctx


def load_schema_metadata(path: Path | None = None) -> dict[str, Any] | None:
    settings = get_settings()
    p = path or settings.metadata.schema_file
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_schema_metadata(metadata: dict[str, Any], path: Path | None = None) -> None:
    settings = get_settings()
    p = path or settings.metadata.schema_file
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
    if not force and cached is not None and cached.get("schema_hash") == live_hash:
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
            "description": info["description"],
            "row_count": info["row_count"],
        }
        for name, info in sorted(metadata.get("tables", {}).items())
    ]
