"""Structured, centralized logging setup for the whole backend."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from app.config import get_settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_configured = False


def configure_logging() -> None:
    """Idempotently configure root logging handlers (console + rotating file)."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    try:
        settings.logging.file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.logging.file, maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError:
        # File logging is a nice-to-have; never let it block startup.
        root.warning("Could not initialize file logging at %s", settings.logging.file)

    # The Azure SDK's HTTP logging policy dumps full request/response headers
    # at INFO level, which drowns out our own logs. Keep it at WARNING.
    for noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
