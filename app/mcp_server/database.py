"""Read-only database MCP server used by data and knowledge workflows.

The server is connected through the MCP SDK's in-memory transport.  This is a
real MCP client/server protocol boundary, but avoids starting a subprocess for
each graph node.  Every public tool delegates to the same deterministic,
allowlisted database implementation used elsewhere in the application.
"""

from __future__ import annotations

from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from app.config import get_settings
from app.db.connection import get_active_database_identity, readonly_connection
from app.metadata import store
from app.sql.validator import validate_sql
from app.sql.time_context import (
    resolve_relative_time_context,
    validate_relative_time_sql,
)
from app.tools.database import call_database_tool


database_mcp = FastMCP(
    "ai-analyst-database",
    instructions=(
        "Read-only SQLite analytics and RAG retrieval. SQL must be validated "
        "before execution; mutation and arbitrary filesystem tools are unavailable."
    ),
)


def _metadata() -> dict:
    cached = store.load_schema_metadata()
    if store.cached_metadata_matches_active_source(cached):
        return cached
    with readonly_connection() as connection:
        metadata, _rebuilt = store.refresh_if_needed(connection)
    return metadata


@database_mcp.tool()
def search_schema(
    question: str,
    hinted_tables: list[str] | None = None,
    top_k: int = 6,
) -> dict:
    """RAG-select the schema slice needed to answer a natural-language query."""
    invocation = call_database_tool(
        "search_schema",
        {
            "question": question,
            "hinted_tables": hinted_tables or [],
            "top_k": top_k,
        },
        stage="mcp_server_schema_retrieval",
        metadata=_metadata(),
    )
    return invocation.value


@database_mcp.tool()
def search_knowledge_documents(question: str, top_k: int = 4) -> dict:
    """RAG-search generated schema and business knowledge documents."""
    invocation = call_database_tool(
        "search_knowledge_documents",
        {"question": question, "top_k": top_k},
        stage="mcp_server_knowledge_retrieval",
        metadata=_metadata(),
    )
    value = invocation.value
    return {
        "mode": value["mode"],
        "documents": [asdict(document) for document in value["documents"]],
    }


@database_mcp.tool()
def resolve_relative_time(question: str, table_names: list[str]) -> dict:
    """Resolve relative time wording to one shared database-aware date range."""
    return resolve_relative_time_context(question, _metadata(), table_names)


@database_mcp.tool()
def validate_readonly_sql(
    sql: str,
    allowed_tables: list[str] | None = None,
    time_context: dict | None = None,
    question: str = "",
) -> dict:
    """Enforce read-only, live-schema, type, join, time, and intent policies."""
    metadata = _metadata()
    invocation = call_database_tool(
        "validate_sql",
        {"sql": sql, "question": question},
        stage="mcp_server_sql_validation",
        metadata={
            **metadata,
            "tables": {
                name: table
                for name, table in metadata.get("tables", {}).items()
                if not allowed_tables or name in set(allowed_tables)
            },
        },
    )
    result = invocation.value
    if result.is_valid and time_context:
        semantic_errors = validate_relative_time_sql(
            result.sanitized_sql or sql, time_context
        )
        if semantic_errors:
            result.is_valid = False
            result.errors.extend(semantic_errors)
            result.sanitized_sql = None
            result.download_sql = None
    return asdict(result)


@database_mcp.tool()
def execute_readonly_sql(
    sql: str,
    max_rows: int | None = None,
    timeout_seconds: int | None = None,
) -> dict:
    """Execute already validated SQL against the active query-only connection."""
    settings = get_settings()
    effective_max_rows = max_rows or settings.limits.max_rows
    # MCP calls are independent requests, so never trust a client to have made
    # the validation call first. Revalidate at the execution boundary as well.
    metadata = _metadata()
    validation = validate_sql(
        sql,
        set(metadata.get("tables", {})),
        effective_max_rows,
        download_max_rows=effective_max_rows,
    )
    if not validation.is_valid or not validation.sanitized_sql:
        raise ValueError(
            "SQL was blocked by the read-only MCP policy: "
            + "; ".join(validation.errors or ["validation did not produce SQL"])
        )
    invocation = call_database_tool(
        "execute_readonly_sql",
        {
            "sql": validation.sanitized_sql,
            "max_rows": effective_max_rows,
            "timeout_seconds": (
                timeout_seconds or settings.limits.statement_timeout_seconds
            ),
        },
        stage="mcp_server_query_execution",
    )
    result = invocation.value
    return {
        "success": result.success,
        "error": result.error,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "duration_ms": result.duration_ms,
        # JSON split is compact, preserves column order, and does not expose a
        # Python/pickle deserialization surface across the protocol boundary.
        "dataframe_json": (
            result.dataframe.to_json(
                orient="split",
                date_format="iso",
                double_precision=15,
                default_handler=str,
            )
            if result.dataframe is not None
            else None
        ),
    }


@database_mcp.tool()
def get_database_rag_status() -> dict:
    """Report the active database and its generated/vector knowledge status."""
    identity = get_active_database_identity()
    from app.metadata import vector_store

    return {"database_identity": identity, **vector_store.collection_stats(identity)}


if __name__ == "__main__":  # pragma: no cover - useful for external MCP clients
    database_mcp.run(transport="stdio")
