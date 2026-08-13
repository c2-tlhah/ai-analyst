"""Low-latency MCP gateway for database and knowledge-base operations."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import timedelta
from io import StringIO
from typing import Any

import pandas as pd

from app.config import get_settings
from app.db.executor import ExecutionResult
from app.metadata.vector_store import RetrievedDocument
from app.mcp_server.database import database_mcp
from app.observability import emit_trace
from app.sql.validator import ValidationResult
from app.tools.database import (
    DatabaseToolError,
    ToolCallRecord,
    ToolInvocation,
)


DATABASE_MCP_TOOLS = frozenset(
    {
        "search_schema",
        "search_knowledge_documents",
        "resolve_relative_time",
        "validate_readonly_sql",
        "execute_readonly_sql",
        "get_database_rag_status",
    }
)


def _tool_text(response: Any) -> str:
    blocks = []
    for content in response.content or []:
        if hasattr(content, "text"):
            blocks.append(str(content.text))
    return "\n".join(blocks)


async def _call_async(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from mcp.shared.memory import create_connected_server_and_client_session

    timeout = max(1, get_settings().limits.statement_timeout_seconds + 10)
    async with create_connected_server_and_client_session(
        database_mcp,
        read_timeout_seconds=timedelta(seconds=timeout),
        raise_exceptions=True,
    ) as session:
        available = {tool.name for tool in (await session.list_tools()).tools}
        if name not in DATABASE_MCP_TOOLS or name not in available:
            raise DatabaseToolError(f"Blocked unknown database MCP tool: {name!r}.")
        response = await session.call_tool(name, arguments=arguments)
        if response.isError:
            raise DatabaseToolError(
                _tool_text(response) or f"Database MCP tool {name} failed."
            )
        payload = response.structuredContent
        if payload is None:
            raw = _tool_text(response)
            if not raw:
                return {}
            payload = json.loads(raw)
        # FastMCP may wrap non-object returns, but every database tool is defined
        # to return an object. Keep a defensive unwrap for SDK compatibility.
        if isinstance(payload, dict) and set(payload) == {"result"}:
            payload = payload["result"]
        if not isinstance(payload, dict):
            raise DatabaseToolError(
                f"Database MCP tool {name} returned a non-object response."
            )
        return payload


def _run(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        return asyncio.run(_call_async(name, arguments))
    except TimeoutError as exc:
        raise DatabaseToolError(f"Database MCP tool {name} timed out.") from exc
    except DatabaseToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize protocol errors
        raise DatabaseToolError(f"Database MCP tool {name} failed: {exc}") from exc


def _decode(name: str, payload: dict[str, Any]) -> Any:
    if name == "validate_readonly_sql":
        return ValidationResult(**payload)
    if name == "execute_readonly_sql":
        dataframe_json = payload.pop("dataframe_json", None)
        dataframe = (
            pd.read_json(StringIO(dataframe_json), orient="split")
            if dataframe_json
            else None
        )
        return ExecutionResult(dataframe=dataframe, **payload)
    if name == "search_knowledge_documents":
        return {
            "mode": payload.get("mode", "lexical"),
            "documents": [
                RetrievedDocument(**document)
                for document in payload.get("documents", [])
            ],
        }
    return payload


def _summary(name: str, value: Any) -> str:
    if name == "search_schema":
        count = len(value.get("metadata", {}).get("tables", {}))
        return f"MCP + RAG selected {count} relevant table(s) using {value.get('mode')} retrieval."
    if name == "search_knowledge_documents":
        return (
            f"MCP + RAG retrieved {len(value.get('documents', []))} knowledge "
            f"document(s) using {value.get('mode')} search."
        )
    if name == "resolve_relative_time":
        return str(value.get("label") or "MCP resolved the relative time period.")
    if name == "validate_readonly_sql":
        if value.is_valid and value.repairs:
            return "MCP validated and safely normalized SQL: " + " ".join(value.repairs)
        return "MCP read-only SQL validation passed." if value.is_valid else (
            f"MCP SQL validation found {len(value.errors)} issue(s)."
        )
    if name == "execute_readonly_sql":
        return (
            f"MCP read-only execution returned {value.row_count} row(s)."
            if value.success
            else "MCP read-only execution failed."
        )
    if name == "get_database_rag_status":
        return "MCP inspected database RAG status."
    return f"Database MCP tool {name} completed."


def call_database_mcp_tool(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    stage: str,
) -> ToolInvocation:
    """Call one allowlisted database MCP tool and return a UI audit record."""
    arguments = dict(arguments or {})
    started = time.monotonic()
    call_id = f"dbmcp-{uuid.uuid4().hex[:12]}"
    safe_arguments = {
        key: (str(value)[:2000] if key == "sql" else value)
        for key, value in arguments.items()
    }
    emit_trace(
        name,
        category="mcp_database_tool",
        status="started",
        metadata={"call_id": call_id, "stage": stage, "arguments": safe_arguments},
    )
    try:
        payload = _run(name, arguments)
        value = _decode(name, payload)
        record = ToolCallRecord(
            call_id=call_id,
            name=name,
            stage=stage,
            status="completed",
            summary=_summary(name, value),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            arguments=safe_arguments,
            transport="mcp",
        )
        emit_trace(
            name,
            category="mcp_database_tool",
            status="completed",
            duration_ms=record.duration_ms,
            message=record.summary,
            metadata={"call_id": call_id, "stage": stage, "transport": "memory"},
        )
        return ToolInvocation(value=value, record=record)
    except Exception as exc:
        record = ToolCallRecord(
            call_id=call_id,
            name=name,
            stage=stage,
            status="failed",
            summary=f"Database MCP tool {name} failed.",
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            arguments=safe_arguments,
            error=str(exc),
            transport="mcp",
        )
        emit_trace(
            name,
            category="mcp_database_tool",
            status="failed",
            duration_ms=record.duration_ms,
            message=str(exc),
            metadata={"call_id": call_id, "stage": stage, "transport": "memory"},
        )
        error = DatabaseToolError(str(exc))
        error.tool_record = record  # type: ignore[attr-defined]
        raise error from exc
