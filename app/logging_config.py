"""Structured, centralized and secret-safe logging for the backend."""

from __future__ import annotations

import logging
import re
import sys
import json
from logging.handlers import RotatingFileHandler

from app.config import get_settings

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | trace=%(trace_id)s | %(message)s"
)
TRACE_LOGGER_NAME = "ai_analyst.agent_trace"

_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(
        r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)"
        r"([^\s,;]+)"
    ),
    re.compile(r"\b(?:nvapi|sk-or-v1|sk)-[A-Za-z0-9._-]{8,}\b"),
)

_configured = False


def redact_log_text(value: object) -> str:
    """Remove common credential forms before text reaches a log handler."""
    text = str(value)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups:
            text = pattern.sub(r"\1[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


class _ContextAndRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from app.observability import current_trace_id

            record.trace_id = current_trace_id() or "-"
        except Exception:  # pragma: no cover - logging must never break startup
            record.trace_id = "-"
        record.msg = redact_log_text(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: redact_log_text(value) if isinstance(value, str) else value
                    for key, value in record.args.items()
                }
            else:
                record.args = tuple(
                    redact_log_text(value) if isinstance(value, str) else value
                    for value in record.args
                )
        return True


class _TraceJSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "trace_event", None)
        if not isinstance(event, dict):
            event = {
                "timestamp": self.formatTime(record),
                "trace_id": getattr(record, "trace_id", "-"),
                "category": "log",
                "name": record.name,
                "status": record.levelname.casefold(),
                "message": record.getMessage(),
            }
        return json.dumps(event, ensure_ascii=False, default=str, separators=(",", ":"))


class _SafeTextFormatter(logging.Formatter):
    """Redact the final rendered line, including exception/traceback text."""

    def format(self, record: logging.LogRecord) -> str:
        return redact_log_text(super().format(record))


def configure_logging() -> None:
    """Idempotently configure root logging handlers (console + rotating file)."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = _SafeTextFormatter(_LOG_FORMAT)
    safe_filter = _ContextAndRedactionFilter()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(safe_filter)
    root.addHandler(console_handler)

    try:
        settings.logging.file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.logging.file,
            maxBytes=max(1, settings.logging.max_bytes),
            backupCount=max(1, settings.logging.backup_count),
            encoding="utf-8",
            delay=True,
        )
        file_handler.setFormatter(formatter)
        file_handler.addFilter(safe_filter)
        root.addHandler(file_handler)
    except OSError:
        # File logging is a nice-to-have; never let it block startup.
        root.warning("Could not initialize file logging at %s", settings.logging.file)

    try:
        settings.logging.trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_handler = RotatingFileHandler(
            settings.logging.trace_file,
            maxBytes=max(1, settings.logging.trace_max_bytes),
            backupCount=max(1, settings.logging.trace_backup_count),
            encoding="utf-8",
            delay=True,
        )
        trace_handler.setFormatter(_TraceJSONFormatter())
        trace_handler.addFilter(safe_filter)
        trace_logger = logging.getLogger(TRACE_LOGGER_NAME)
        trace_logger.setLevel(level)
        trace_logger.addHandler(trace_handler)
        # Trace events also propagate to the readable application log/console.
        trace_logger.propagate = True
    except OSError:
        root.warning(
            "Could not initialize agent trace logging at %s",
            settings.logging.trace_file,
        )

    # The Azure SDK's HTTP logging policy dumps full request/response headers
    # at INFO level, which drowns out our own logs. Keep it at WARNING.
    for noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
