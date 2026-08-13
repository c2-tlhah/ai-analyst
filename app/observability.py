"""End-to-end agent tracing with safe in-memory and JSONL event stores.

The trace is deliberately provider-neutral. Database tools, LLM stages, MCP
operations, catalog discovery, and top-level requests all emit the same small
event shape. Metadata is recursively redacted before it is retained or logged.
"""

from __future__ import annotations

import contextvars
import functools
import json
import re
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, TypeVar

from app.config import get_settings
from app.logging_config import TRACE_LOGGER_NAME, get_logger, redact_log_text

logger = get_logger(TRACE_LOGGER_NAME)

_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_trace_id", default=None
)
_span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agent_span_id", default=None
)
_events_lock = threading.Lock()
_events: deque[dict[str, Any]] | None = None
_SECRET_KEY = re.compile(
    r"(?i)(api[_-]?key|authorization|credential|password|secret|token)"
)
_MAX_TEXT = 2000
F = TypeVar("F", bound=Callable[..., Any])


def current_trace_id() -> str | None:
    return _trace_id_var.get()


def _event_buffer() -> deque[dict[str, Any]]:
    global _events
    if _events is None:
        max_events = max(100, get_settings().logging.trace_memory_events)
        loaded: deque[dict[str, Any]] = deque(maxlen=max_events)
        trace_path = get_settings().logging.trace_file
        try:
            with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if isinstance(event, dict):
                        loaded.append(event)
        except FileNotFoundError:
            pass
        _events = loaded
    return _events


def _safe(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if depth > 5:
        return "[MAX_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return redact_log_text(value)[:_MAX_TEXT]
    if isinstance(value, dict):
        return {
            str(item_key)[:100]: _safe(item_value, key=str(item_key), depth=depth + 1)
            for item_key, item_value in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe(item, depth=depth + 1) for item in list(value)[:100]]
    return redact_log_text(value)[:_MAX_TEXT]


def emit_trace(
    name: str,
    *,
    category: str,
    status: str,
    message: str = "",
    duration_ms: int | None = None,
    metadata: dict[str, Any] | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
) -> dict[str, Any]:
    """Persist and publish one presentation-safe trace event."""
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "trace_id": trace_id or current_trace_id() or f"trace-{uuid.uuid4().hex[:16]}",
        "span_id": span_id or _span_id_var.get() or f"span-{uuid.uuid4().hex[:12]}",
        "parent_span_id": parent_span_id,
        "category": str(category),
        "name": str(name),
        "status": str(status),
        "duration_ms": duration_ms,
        "message": redact_log_text(message)[:_MAX_TEXT],
        "metadata": _safe(metadata or {}),
    }
    with _events_lock:
        _event_buffer().append(event)
    level = "error" if status == "failed" else "warning" if status == "retrying" else "info"
    getattr(logger, level)(
        "%s.%s %s%s",
        category,
        name,
        status,
        f": {event['message']}" if event["message"] else "",
        extra={"trace_event": event},
    )
    return event


@contextmanager
def trace_span(
    name: str,
    *,
    category: str = "agent",
    metadata: dict[str, Any] | None = None,
) -> Iterator[str]:
    """Create a timed span, automatically recording completion or failure."""
    existing_trace = current_trace_id()
    trace_id = existing_trace or f"trace-{uuid.uuid4().hex[:16]}"
    parent_span_id = _span_id_var.get()
    span_id = f"span-{uuid.uuid4().hex[:12]}"
    trace_token = _trace_id_var.set(trace_id)
    span_token = _span_id_var.set(span_id)
    started = time.monotonic()
    emit_trace(
        name,
        category=category,
        status="started",
        metadata=metadata,
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
    )
    try:
        yield trace_id
    except Exception as exc:
        emit_trace(
            name,
            category=category,
            status="failed",
            message=str(exc),
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            metadata={"error_type": type(exc).__name__},
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
        raise
    else:
        emit_trace(
            name,
            category=category,
            status="completed",
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
        )
    finally:
        _span_id_var.reset(span_token)
        _trace_id_var.reset(trace_token)


def traced_operation(
    name: str,
    *,
    category: str = "agent",
) -> Callable[[F], F]:
    """Trace a public operation and attach its trace ID to mutable responses."""
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with trace_span(name, category=category) as trace_id:
                result = func(*args, **kwargs)
                if hasattr(result, "trace_id"):
                    try:
                        result.trace_id = trace_id
                    except (AttributeError, TypeError):
                        try:
                            object.__setattr__(result, "trace_id", trace_id)
                        except (AttributeError, TypeError):
                            pass
                result_failed = (
                    getattr(result, "status", None) == "error"
                    or getattr(result, "success", True) is False
                )
                if result_failed:
                    emit_trace(
                        "handled_result",
                        category=category,
                        status="failed",
                        message=str(
                            getattr(result, "error", None)
                            or getattr(result, "message", None)
                            or "Operation returned an error response."
                        ),
                    )
                return result

        return wrapper  # type: ignore[return-value]

    return decorator


def get_recent_trace_events(
    *, limit: int = 200, trace_id: str | None = None
) -> list[dict[str, Any]]:
    with _events_lock:
        items = list(_event_buffer())
    if trace_id:
        items = [item for item in items if item.get("trace_id") == trace_id]
    return items[-max(1, min(limit, 2000)):]


def read_log_tail(path: Path, *, max_lines: int = 500) -> str:
    """Read a bounded, redacted tail for the UI without exposing arbitrary files."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    return "\n".join(redact_log_text(line) for line in lines[-max(1, max_lines):])


def export_recent_traces(*, limit: int = 2000) -> bytes:
    return (
        "\n".join(
            json.dumps(event, ensure_ascii=False, default=str)
            for event in get_recent_trace_events(limit=limit)
        )
        + "\n"
    ).encode("utf-8")
