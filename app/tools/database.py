"""Small, secure database tool registry.

The application has two kinds of reasoning:

* deterministic operations that must always be safe and reproducible (connect,
  inspect, validate, execute); and
* language-model operations that interpret or describe those facts.

This module gives the deterministic operations one typed, auditable interface.
The same calls are used by the connection workflow and the LangGraph query
workflow, so the UI can show exactly what happened without exposing raw
connections or allowing models to execute arbitrary Python.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db.connection import (
    get_active_database_identity,
    get_active_database_path,
    readonly_connection,
    set_active_database_path,
    validate_database_source,
)
from app.db.executor import execute_sql
from app.llm.client import LLMClient
from app.metadata import discovery, enrichment, retrieval, store, vector_store
from app.observability import emit_trace
from app.sql.validator import validate_sql


class DatabaseToolError(RuntimeError):
    """Raised when an approved database tool cannot complete safely."""


@dataclass(frozen=True)
class ToolCallRecord:
    """Presentation-safe audit record for one backend tool invocation."""

    call_id: str
    name: str
    stage: str
    status: str
    summary: str
    duration_ms: int
    arguments: dict[str, Any]
    error: str | None = None
    transport: str = "internal"

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "name": self.name,
            "stage": self.stage,
            "status": self.status,
            "summary": self.summary,
            "duration_ms": self.duration_ms,
            "arguments": self.arguments,
            "error": self.error,
            "transport": self.transport,
        }


@dataclass(frozen=True)
class ToolInvocation:
    """A tool's internal value plus its user-safe audit record."""

    value: Any
    record: ToolCallRecord


def _definition(name: str, description: str, properties: dict[str, Any], required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


# These JSON schemas document the contract and can also be supplied to a
# tool-capable LLM in future workflows. Execution always remains behind
# ``call_database_tool`` and its closed dispatcher.
DATABASE_TOOL_DEFINITIONS = (
    _definition(
        "connect_database",
        "Validate and activate one existing SQLite database in read-only mode.",
        {"source": {"type": "string"}},
        ("source",),
    ),
    _definition("list_databases", "List SQLite databases attached to the read-only connection.", {}),
    _definition(
        "get_database_info",
        "Return read-only identity, file size, page size, and SQLite user version.",
        {},
    ),
    _definition("list_tables", "List user tables in the active SQLite database.", {}),
    _definition(
        "inspect_database_schema",
        "Inspect all tables, columns, profiles, keys, and relationships in one bounded call.",
        {},
    ),
    _definition(
        "get_table_schema",
        "Inspect columns, keys, types, semantic roles, and safe categorical samples.",
        {"table": {"type": "string"}},
        ("table",),
    ),
    _definition("get_relationships", "List declared foreign-key relationships.", {}),
    _definition(
        "sample_table",
        "Read a small bounded sample from an approved table.",
        {
            "table": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ("table",),
    ),
    _definition(
        "profile_columns",
        "Return deterministic roles, cardinalities, and aggregation hints for a table.",
        {"table": {"type": "string"}},
        ("table",),
    ),
    _definition(
        "generate_descriptions",
        "Generate or reuse business descriptions for discovered tables and columns.",
        {},
    ),
    _definition(
        "generate_knowledge_documents",
        "Render one complete documentation file per discovered table.",
        {},
    ),
    _definition(
        "write_knowledge_documents",
        "Version and save generated documentation, then optionally create a semantic index.",
        {},
    ),
    _definition(
        "search_schema",
        "Select relevant schema documents for a natural-language data question.",
        {
            "question": {"type": "string"},
            "hinted_tables": {"type": "array", "items": {"type": "string"}},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ("question",),
    ),
    _definition(
        "search_knowledge_documents",
        "Search generated database documentation, with vector search when available and lexical fallback.",
        {
            "question": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        ("question",),
    ),
    _definition(
        "validate_sql",
        "Parse SQL and enforce read-only, live-schema, type, join, and question-alignment policies.",
        {
            "sql": {"type": "string"},
            "question": {"type": "string"},
        },
        ("sql",),
    ),
    _definition(
        "execute_readonly_sql",
        "Execute previously validated SQL through a read-only SQLite connection.",
        {"sql": {"type": "string"}},
        ("sql",),
    ),
)


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [str(row["name"]) for row in rows]


def _canonical_table(conn: sqlite3.Connection, requested: str) -> str:
    matches = {name.casefold(): name for name in _user_tables(conn)}
    canonical = matches.get(str(requested or "").strip().casefold())
    if not canonical:
        raise DatabaseToolError(f"Unknown or unavailable table: {requested!r}.")
    return canonical


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _json_safe_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep records serializable and bounded without storing returned row data."""
    safe: dict[str, Any] = {}
    for key, value in arguments.items():
        if key == "metadata" or key == "documents" or key == "llm_client":
            continue
        if key == "sql":
            safe[key] = str(value)[:2000]
        elif isinstance(value, Path):
            safe[key] = str(value)
        else:
            try:
                json.dumps(value, default=str)
                safe[key] = value
            except TypeError:
                safe[key] = str(value)
    return safe


def _summarize(name: str, value: Any) -> str:
    if name == "connect_database":
        return f"Connected read-only to {Path(value['path']).name}."
    if name == "list_databases":
        return f"Found {len(value)} attached database(s)."
    if name == "get_database_info":
        return f"Inspected {Path(value['path']).name} ({value['size_bytes']} bytes)."
    if name == "list_tables":
        return f"Found {len(value)} user table(s)."
    if name == "inspect_database_schema":
        return (
            f"Inspected {len(value['tables'])} table(s), "
            f"{value['column_count']} column(s), and "
            f"{len(value['relationships'])} relationship(s) in one call."
        )
    if name == "get_table_schema":
        return f"Inspected {value['name']} ({len(value['columns'])} columns)."
    if name == "get_relationships":
        return f"Found {len(value)} foreign-key relationship(s)."
    if name == "sample_table":
        return f"Sampled {len(value['rows'])} row(s) from {value['table']}."
    if name == "profile_columns":
        return f"Profiled {len(value['columns'])} column(s) in {value['table']}."
    if name == "generate_descriptions":
        return f"Generated or reused descriptions for {len(value.get('tables', {}))} table(s)."
    if name == "generate_knowledge_documents":
        return f"Generated {len(value)} knowledge document(s)."
    if name == "write_knowledge_documents":
        status = value.get("status", "ready")
        writer = value.get("document_writer", "managed_local")
        summary = (
            f"Saved {value.get('document_count', 0)} document(s) via {writer}; "
            f"semantic index status: {status}."
        )
        if value.get("mcp_error"):
            summary += f" Filesystem MCP fell back safely: {value['mcp_error']}"
        return summary[:1000]
    if name == "search_schema":
        return f"Selected {len(value['metadata'].get('tables', {}))} relevant table(s) using {value['mode']} retrieval."
    if name == "search_knowledge_documents":
        return f"Retrieved {len(value['documents'])} document(s) using {value['mode']} search."
    if name == "validate_sql":
        return "SQL validation passed." if value.is_valid else f"SQL validation found {len(value.errors)} issue(s)."
    if name == "execute_readonly_sql":
        return f"Read-only query returned {value.row_count} row(s)." if value.success else "Read-only query failed."
    return f"{name} completed."


def _dispatch(
    name: str,
    arguments: dict[str, Any],
    *,
    metadata: dict[str, Any] | None,
    llm_client: LLMClient | None,
    documents: dict[str, str] | None,
) -> Any:
    if name == "connect_database":
        path = validate_database_source(str(arguments.get("source") or ""))
        set_active_database_path(path)
        # Opening once proves SQLite can read the file before state is committed
        # to later discovery steps.
        with readonly_connection(path) as conn:
            conn.execute("SELECT 1").fetchone()
        return {"path": str(path), "identity": get_active_database_identity()}

    if name == "list_databases":
        with readonly_connection() as conn:
            return [
                {"sequence": int(row[0]), "name": str(row[1]), "path": str(row[2] or "")}
                for row in conn.execute("PRAGMA database_list").fetchall()
            ]

    if name == "get_database_info":
        path = get_active_database_path()
        with readonly_connection() as conn:
            page_count = int(conn.execute("PRAGMA page_count").fetchone()[0])
            page_size = int(conn.execute("PRAGMA page_size").fetchone()[0])
            user_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        return {
            "path": str(path),
            "identity": get_active_database_identity(),
            "size_bytes": int(path.stat().st_size),
            "page_count": page_count,
            "page_size": page_size,
            "user_version": user_version,
        }

    if name == "list_tables":
        with readonly_connection() as conn:
            return _user_tables(conn)

    if name == "inspect_database_schema":
        with readonly_connection() as conn:
            discovered = discovery.discover_schema(conn)
        schemas = {name: info.to_dict() for name, info in discovered.items()}
        relationships = [
            {
                "from_table": table_name,
                "from_column": column.name,
                "to_table": column.references["table"],
                "to_column": column.references["column"],
                "source": column.relationship_source or "declared",
                "constraint_id": column.relationship_constraint_id,
                "constraint_sequence": column.relationship_constraint_sequence,
                "constraint_size": column.relationship_constraint_size,
            }
            for table_name, table in discovered.items()
            for column in table.columns
            if column.references
        ]
        return {
            "tables": list(discovered),
            "schemas": schemas,
            "profiles": {
                table_name: {
                    "row_count": table.row_count,
                    "columns": [
                        {
                            "name": column.name,
                            "semantic_role": column.semantic_role,
                            "distinct_count": column.distinct_count,
                            "default_aggregation": column.default_aggregation,
                        }
                        for column in table.columns
                    ],
                }
                for table_name, table in discovered.items()
            },
            "relationships": relationships,
            "column_count": sum(
                len(table.columns) for table in discovered.values()
            ),
        }

    if name == "get_table_schema":
        with readonly_connection() as conn:
            table = _canonical_table(conn, str(arguments.get("table") or ""))
            return discovery.discover_table(conn, table).to_dict()

    if name == "get_relationships":
        if metadata is not None:
            return list(metadata.get("relationships", []))
        with readonly_connection() as conn:
            tables = discovery.discover_schema(conn)
        return [
            {
                "from_table": table.name,
                "from_column": column.name,
                "to_table": column.references["table"],
                "to_column": column.references["column"],
                "constraint_id": column.relationship_constraint_id,
                "constraint_sequence": column.relationship_constraint_sequence,
                "constraint_size": column.relationship_constraint_size,
            }
            for table in tables.values()
            for column in table.columns
            if column.references
        ]

    if name == "sample_table":
        limit = min(max(int(arguments.get("limit", 5)), 1), 20)
        with readonly_connection() as conn:
            table = _canonical_table(conn, str(arguments.get("table") or ""))
            rows = conn.execute(
                f"SELECT * FROM {_quote_identifier(table)} LIMIT ?", (limit,)
            ).fetchall()
            return {"table": table, "rows": [dict(row) for row in rows]}

    if name == "profile_columns":
        with readonly_connection() as conn:
            table = _canonical_table(conn, str(arguments.get("table") or ""))
            info = discovery.discover_table(conn, table)
        return {
            "table": table,
            "row_count": info.row_count,
            "columns": [
                {
                    "name": column.name,
                    "semantic_role": column.semantic_role,
                    "distinct_count": column.distinct_count,
                    "default_aggregation": column.default_aggregation,
                }
                for column in info.columns
            ],
        }

    if name == "generate_descriptions":
        enrich_fn = enrichment.make_llm_enrich_fn(llm_client) if llm_client else None
        with readonly_connection() as conn:
            built, _ = store.refresh_if_needed(conn, enrich_fn=enrich_fn, force=True)
        return built

    if name == "generate_knowledge_documents":
        if metadata is None:
            raise DatabaseToolError("Metadata is required to generate knowledge documents.")
        return vector_store.render_documents(metadata)

    if name == "write_knowledge_documents":
        if metadata is None:
            raise DatabaseToolError("Metadata is required to save knowledge documents.")
        version = vector_store.sync_collection(
            metadata,
            db_identity=get_active_database_identity(),
            documents=documents,
        )
        stats = vector_store.collection_stats(get_active_database_identity())
        doc_count = len(documents or {})
        writer = "managed_local"
        mcp_error = None
        mcp_tool_count = 0
        mcp_config = get_settings().mcp_filesystem
        version_number = int(stats.get("version") or (version or {}).get("version") or 0)
        if (
            documents
            and version_number
            and mcp_config.is_configured
            and mcp_config.allow_mutations
        ):
            try:
                from app.mcp_client.filesystem import write_managed_text_files

                files = vector_store.knowledge_document_files(
                    get_active_database_identity(), version_number, documents
                )
                managed_root = next(iter(files)).parent
                mcp_records = write_managed_text_files(
                    files,
                    managed_root=managed_root,
                    config=mcp_config,
                )
                writer = "filesystem_mcp"
                mcp_tool_count = len(mcp_records)
            except Exception as exc:  # noqa: BLE001 - local managed files already exist
                mcp_error = str(exc)
        return {
            **stats,
            "document_count": max(doc_count, int(stats.get("document_count", 0))),
            "version_record": version,
            "document_writer": writer,
            "mcp_tool_count": mcp_tool_count,
            "mcp_error": mcp_error,
        }

    if name == "search_schema":
        if metadata is None:
            raise DatabaseToolError("Metadata is required to search the schema.")
        settings = get_settings()
        relevant, mode = retrieval.get_relevant_metadata_with_mode(
            metadata,
            str(arguments.get("question") or ""),
            hinted_tables=list(arguments.get("hinted_tables") or []),
            db_identity=get_active_database_identity() if settings.vector.enabled else None,
            top_k=min(max(int(arguments.get("top_k", settings.vector.top_k)), 1), 20),
        )
        ambiguous = (
            mode == "lexical"
            and len(metadata.get("tables", {})) > retrieval.MAX_RELEVANT_TABLES
            and not retrieval.has_lexical_schema_signal(
                metadata, str(arguments.get("question") or "")
            )
        )
        return {"metadata": relevant, "mode": mode, "ambiguous": ambiguous}

    if name == "search_knowledge_documents":
        settings = get_settings()
        hits, mode = vector_store.query_documents_with_fallback(
            str(arguments.get("question") or ""),
            db_identity=get_active_database_identity(),
            top_k=min(max(int(arguments.get("top_k", settings.vector.top_k)), 1), 20),
            fallback_documents=(
                vector_store.render_documents(metadata) if metadata is not None else None
            ),
        )
        return {"documents": hits, "mode": mode}

    if name == "validate_sql":
        if metadata is None:
            raise DatabaseToolError("Metadata is required to validate SQL.")
        settings = get_settings()
        allowed_tables = set(metadata.get("tables", {}).keys())
        return validate_sql(
            str(arguments.get("sql") or ""),
            allowed_tables,
            settings.limits.max_rows,
            download_max_rows=settings.limits.download_max_rows,
            metadata=metadata,
            question=str(arguments.get("question") or ""),
        )

    if name == "execute_readonly_sql":
        settings = get_settings()
        return execute_sql(
            str(arguments.get("sql") or ""),
            max_rows=int(arguments.get("max_rows") or settings.limits.max_rows),
            timeout_seconds=int(
                arguments.get("timeout_seconds")
                or settings.limits.statement_timeout_seconds
            ),
        )

    raise DatabaseToolError(f"Blocked unknown database tool: {name!r}.")


def call_database_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    stage: str,
    metadata: dict[str, Any] | None = None,
    llm_client: LLMClient | None = None,
    documents: dict[str, str] | None = None,
) -> ToolInvocation:
    """Execute one allowlisted tool and always return/raise with an audit record.

    Tool failures are raised as :class:`DatabaseToolError` with the record
    attached as ``tool_record``. This lets orchestrators retain a complete
    activity trail even when a later stage stops the workflow.
    """
    arguments = dict(arguments or {})
    started = time.monotonic()
    call_id = f"dbtool-{uuid.uuid4().hex[:12]}"
    safe_arguments = _json_safe_arguments(arguments)
    emit_trace(
        name,
        category="database_tool",
        status="started",
        metadata={
            "call_id": call_id,
            "stage": stage,
            "arguments": safe_arguments,
        },
    )
    try:
        value = _dispatch(
            name,
            arguments,
            metadata=metadata,
            llm_client=llm_client,
            documents=documents,
        )
        record = ToolCallRecord(
            call_id=call_id,
            name=name,
            stage=stage,
            status="completed",
            summary=_summarize(name, value),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            arguments=safe_arguments,
        )
        emit_trace(
            name,
            category="database_tool",
            status="completed",
            duration_ms=record.duration_ms,
            message=record.summary,
            metadata={"call_id": call_id, "stage": stage},
        )
        return ToolInvocation(value=value, record=record)
    except Exception as exc:  # noqa: BLE001 - normalized for orchestrator/UI handling
        record = ToolCallRecord(
            call_id=call_id,
            name=name,
            stage=stage,
            status="failed",
            summary=f"{name} failed.",
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            arguments=safe_arguments,
            error=str(exc),
        )
        emit_trace(
            name,
            category="database_tool",
            status="failed",
            duration_ms=record.duration_ms,
            message=str(exc),
            metadata={"call_id": call_id, "stage": stage},
        )
        error = DatabaseToolError(str(exc))
        error.tool_record = record  # type: ignore[attr-defined]
        raise error from exc
