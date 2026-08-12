"""Read-only SQLite connection management.

The analytics platform must never be able to mutate the source database.
We enforce that at two independent layers:

1. The OS-level connection is opened with SQLite's ``mode=ro`` URI flag,
   which fails outright on any write attempt.
2. ``PRAGMA query_only = 1`` is set as defense-in-depth in case a future
   driver/DB swap loses the URI flag.

On top of the connection-level guarantee, :mod:`app.sql.validator` adds a
deterministic statement-level allow-list before anything reaches this
connection at all.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseNotFoundError(RuntimeError):
    pass


def _db_uri(path: Path) -> str:
    return f"file:{path.as_posix()}?mode=ro"


def open_readonly_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a brand-new read-only connection to the analytics database."""
    settings = get_settings()
    path = db_path or settings.database.path

    if not path.exists():
        raise DatabaseNotFoundError(
            f"Database not found at {path}. Run `python scripts/build_database.py` first."
        )

    conn = sqlite3.connect(
        _db_uri(path),
        uri=True,
        timeout=settings.limits.statement_timeout_seconds,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = 1;")
    conn.execute(f"PRAGMA busy_timeout = {settings.limits.statement_timeout_seconds * 1000};")
    return conn


@contextmanager
def readonly_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Context-managed read-only connection, closed automatically on exit."""
    conn = open_readonly_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def database_exists(db_path: Path | None = None) -> bool:
    settings = get_settings()
    path = db_path or settings.database.path
    return path.exists()
