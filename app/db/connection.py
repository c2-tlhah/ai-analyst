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

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import PROJECT_ROOT, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseNotFoundError(RuntimeError):
    pass


class InvalidDatabaseSourceError(RuntimeError):
    pass


# AI_ANALYST_DB_PATH supplies only the startup database; the UI's "Connect"
# panel can activate any other SQLite file at
# runtime. This override is process-global (matching the existing
# process-global metadata/answer caches in app.orchestrator) rather than
# per-Streamlit-session -- this app is single-tenant/local by design.
_active_db_path: Path | None = None


def set_active_database_path(path: Path | None) -> None:
    """Switch the database every connection/query resolves to by default."""
    global _active_db_path
    _active_db_path = path


def get_active_database_path() -> Path:
    settings = get_settings()
    return _active_db_path or settings.database.path


def get_active_database_identity() -> str:
    """Stable id for one database file, not merely a reusable filesystem path."""
    path = get_active_database_path()
    canonical = str(path.resolve()) if path.exists() else str(path)
    try:
        stat = path.stat()
        file_identity = f"{stat.st_dev}:{stat.st_ino}"
    except OSError:
        file_identity = "missing"
    return hashlib.sha1(
        f"{canonical}|{file_identity}".encode("utf-8")
    ).hexdigest()[:16]


def get_active_database_revision() -> str:
    """Cheap revision token that changes when database/WAL contents change."""
    path = get_active_database_path()
    parts = [get_active_database_identity()]
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        try:
            stat = candidate.stat()
            parts.append(f"{candidate.name}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            continue
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def resolve_database_source(source: str) -> Path:
    """Normalize a user-supplied filesystem path or SQLite connection string.

    Accepts a plain path (relative paths resolve against the project root,
    matching ``AI_ANALYST_DB_PATH``) or a ``sqlite:///``/``sqlite://``/
    ``file:`` prefixed connection string, optionally with a trailing
    ``?mode=ro``-style query suffix.
    """
    raw = (source or "").strip()
    if not raw:
        raise InvalidDatabaseSourceError("No database path or connection string was provided.")

    for prefix in ("sqlite:///", "sqlite://", "file:"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    raw = raw.split("?", 1)[0].strip()
    if not raw:
        raise InvalidDatabaseSourceError(f"'{source}' does not name a database file.")

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validate_database_source(source: str) -> Path:
    """Resolve ``source`` and confirm it opens as a real, read-only SQLite database.

    Raises :class:`DatabaseNotFoundError` or :class:`InvalidDatabaseSourceError`
    with a user-facing message; returns the resolved path on success.
    """
    path = resolve_database_source(source)
    if not path.exists():
        raise DatabaseNotFoundError(f"No file found at {path}.")

    try:
        conn = open_readonly_connection(path)
        try:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1;")
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise InvalidDatabaseSourceError(
            f"{path} does not look like a valid SQLite database ({exc})."
        ) from exc

    return path


def _db_uri(path: Path) -> str:
    return f"file:{path.as_posix()}?mode=ro"


def open_readonly_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a brand-new read-only connection to the analytics database."""
    settings = get_settings()
    path = db_path or get_active_database_path()

    if not path.exists():
        raise DatabaseNotFoundError(
            f"Database not found at {path}. Connect to an existing SQLite file "
            "from the sidebar or restore the configured database path."
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
