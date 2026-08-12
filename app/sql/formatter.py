"""Cosmetic SQL formatting for display only.

Never used on the execution path -- :mod:`app.sql.validator` already
produced (and the executor already ran) the sanitized statement by the
time this runs. This just makes the *already-safe* SQL readable in the UI
instead of showing the compact single-line form ``sqlglot`` emits after
validation.
"""

from __future__ import annotations

import sqlglot

from app.sql.validator import DIALECT


def format_sql_for_display(sql: str | None) -> str | None:
    if not sql:
        return sql
    try:
        return sqlglot.transpile(sql, read=DIALECT, write=DIALECT, pretty=True)[0]
    except Exception:  # noqa: BLE001 - display formatting must never break the response
        return sql
