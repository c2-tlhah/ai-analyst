"""Deterministic, limited SQL execution against the read-only connection.

This is the only module that actually runs a query. It assumes the SQL it
receives has already passed :func:`app.sql.validator.validate_sql` -- it
adds a belt-and-braces row cap and a wall-clock query timeout (enforced via
SQLite's progress handler, since SQLite has no native statement timeout) on
top of that.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.db.connection import DatabaseNotFoundError, open_readonly_connection
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    dataframe: pd.DataFrame | None = None
    error: str | None = None
    row_count: int = 0
    truncated: bool = False
    duration_ms: float = 0.0


def _install_timeout(conn: sqlite3.Connection, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds

    def handler() -> int:
        return 1 if time.monotonic() > deadline else 0

    # Called every N SQLite VM instructions; returning non-zero aborts the query.
    conn.set_progress_handler(handler, 1000)


def execute_sql(
    sql: str,
    *,
    db_path: Path | None = None,
    max_rows: int | None = None,
    timeout_seconds: int | None = None,
) -> ExecutionResult:
    settings = get_settings()
    max_rows = max_rows or settings.limits.max_rows
    timeout_seconds = timeout_seconds or settings.limits.statement_timeout_seconds

    started = time.monotonic()
    try:
        conn = open_readonly_connection(db_path)
    except DatabaseNotFoundError as exc:
        return ExecutionResult(success=False, error=str(exc))

    try:
        _install_timeout(conn, timeout_seconds)
        df = pd.read_sql_query(sql, conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        message = str(exc)
        if "interrupted" in message.lower():
            logger.warning("Query exceeded timeout of %ss", timeout_seconds)
            return ExecutionResult(
                success=False, error=f"Query exceeded the {timeout_seconds}s execution time limit."
            )
        logger.warning("SQL execution failed: %s", message)
        return ExecutionResult(success=False, error=f"Database error: {message}")
    except Exception as exc:  # noqa: BLE001 - surface as a clean execution error
        logger.exception("Unexpected error executing SQL")
        return ExecutionResult(success=False, error=f"Unexpected execution error: {exc}")
    finally:
        conn.close()

    duration_ms = (time.monotonic() - started) * 1000
    truncated = len(df) >= max_rows
    if truncated:
        df = df.iloc[:max_rows].copy()

    return ExecutionResult(
        success=True,
        dataframe=df,
        row_count=len(df),
        truncated=truncated,
        duration_ms=duration_ms,
    )
