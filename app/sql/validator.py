"""Deterministic SQL security/validation layer.

This is the hard gate between whatever the LLM generates and the database.
Nothing here is probabilistic: every check is a parse-tree inspection or a
whitelist lookup against the metadata store. If a query doesn't pass every
check it is rejected before it ever reaches :mod:`app.sql.executor`.

Checks enforced:
  * exactly one statement (no stacked/multi statements)
  * the statement is a read-only query (``SELECT`` / set operations /
    CTEs) -- anything else (INSERT, UPDATE, DELETE, DDL, PRAGMA, ATTACH,
    ...) is rejected
  * a textual keyword scan as defense-in-depth on top of the parse-tree check
  * every real table referenced is in the caller-supplied allow-list
    (CTE-local names are recognized and excluded from that check)
  * a ``LIMIT`` is enforced, capped at the configured maximum row count
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.logging_config import get_logger

logger = get_logger(__name__)

DIALECT = "sqlite"

# Defense-in-depth textual scan. The parse-tree check (statement must be an
# `exp.Query`) already rejects all of these; this catches anything a future
# sqlglot version might parse differently than expected.
_FORBIDDEN_KEYWORDS = (
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE", "TRUNCATE",
    "REPLACE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
    "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK",
)
_FORBIDDEN_FUNCTIONS = {
    "load_extension",
    "readfile",
    "writefile",
    "eval",
}

_NON_CODE_SQL = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|`(?:``|[^`])*`|\[[^\]]*\]"
    r"|--[^\r\n]*|/\*.*?\*/",
    flags=re.DOTALL,
)


def _code_only_sql(sql: str) -> str:
    """Mask values, quoted identifiers, and comments before text heuristics."""
    return _NON_CODE_SQL.sub(lambda match: " " * len(match.group(0)), sql)


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    sanitized_sql: str | None = None
    download_sql: str | None = None
    tables_referenced: list[str] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)


def _cte_alias_names(parsed: exp.Expression) -> set[str]:
    names: set[str] = set()
    for with_expr in parsed.find_all(exp.With):
        for cte in with_expr.expressions:
            if cte.alias:
                names.add(cte.alias.lower())
    return names


def _enforce_limit(parsed: exp.Expression, max_rows: int) -> exp.Expression:
    existing = parsed.args.get("limit")
    if existing is not None:
        try:
            current = int(existing.expression.this)
        except (TypeError, ValueError, AttributeError):
            current = None
        if current is None or current > max_rows:
            parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    return parsed


def _enforce_download_limit(parsed: exp.Expression, max_rows: int) -> exp.Expression:
    """Preserve a requested LIMIT, or add an export cap plus one sentinel row."""
    max_rows = max(1, max_rows)
    existing = parsed.args.get("limit")
    if existing is not None:
        try:
            current = int(existing.expression.this)
        except (TypeError, ValueError, AttributeError):
            current = None
        if current is not None and current <= max_rows:
            return parsed

    # Fetch one additional row so execution can report when the configurable
    # download safety cap was reached.
    parsed.set("limit", exp.Limit(expression=exp.Literal.number(max_rows + 1)))
    return parsed


def validate_sql(
    sql: str,
    allowed_tables: set[str],
    max_rows: int,
    download_max_rows: int | None = None,
    metadata: dict | None = None,
    question: str = "",
) -> ValidationResult:
    """Validate and sanitize a candidate SQL string.

    ``allowed_tables`` should be the set of tables relevant to the current
    request (or the full catalog); comparison is case-insensitive.
    """
    errors: list[str] = []
    sql = (sql or "").strip()

    if not sql:
        return ValidationResult(is_valid=False, errors=["Empty SQL statement."])

    code_sql = _code_only_sql(sql)
    upper_sql = code_sql.upper()
    for kw in _FORBIDDEN_KEYWORDS:
        if _contains_keyword(upper_sql, kw):
            errors.append(f"Forbidden keyword detected: {kw}.")

    # Give the correction loop actionable SQLite feedback instead of only a
    # low-level parser location when a model drifts into another SQL dialect.
    if re.search(
        r"\bTOP\s*(?:\(\s*\d+\s*\)|\d+)", code_sql, flags=re.IGNORECASE
    ):
        errors.append("SQLite does not support TOP; put LIMIT N at the end of the query.")
    if "::" in code_sql:
        errors.append("SQLite does not support PostgreSQL-style :: syntax.")
    if re.search(r",\s*FROM\b", code_sql, flags=re.IGNORECASE):
        errors.append("Remove the trailing comma immediately before FROM.")

    try:
        statements = [s for s in sqlglot.parse(sql, dialect=DIALECT) if s is not None]
    except ParseError as exc:
        errors.append(f"SQL failed to parse: {exc}")
        return ValidationResult(is_valid=False, errors=errors)

    if len(statements) != 1:
        errors.append(
            f"Exactly one SQL statement is allowed; found {len(statements)}."
        )
        return ValidationResult(is_valid=False, errors=errors)

    parsed = statements[0]

    if not isinstance(parsed, exp.Query):
        errors.append(
            f"Only read-only SELECT queries are allowed; got statement type "
            f"'{type(parsed).__name__}'."
        )
        return ValidationResult(is_valid=False, errors=errors)

    for function in parsed.find_all(exp.Anonymous):
        if function.name.casefold() in _FORBIDDEN_FUNCTIONS:
            errors.append(f"Unsafe SQLite function is not allowed: {function.name}.")

    cte_names = _cte_alias_names(parsed)
    allowed_lower = {t.lower() for t in allowed_tables}
    referenced_tables: list[str] = []
    for table_expr in parsed.find_all(exp.Table):
        name = table_expr.name
        if not name:
            continue
        name_lower = name.lower()
        if name_lower in cte_names:
            continue
        referenced_tables.append(name)
        if name_lower not in allowed_lower:
            errors.append(f"Unauthorized table referenced: {name}.")

    if not referenced_tables:
        errors.append("Query does not reference any table.")

    if errors:
        return ValidationResult(is_valid=False, errors=errors, tables_referenced=referenced_tables)

    repairs: list[str] = []
    if metadata is not None:
        from app.sql.repair import repair_unambiguous_ranking
        from app.sql.semantic_validator import validate_query_semantics

        parsed, repairs = repair_unambiguous_ranking(parsed, question)
        errors.extend(
            validate_query_semantics(parsed, metadata=metadata, question=question)
        )
        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                tables_referenced=referenced_tables,
                repairs=repairs,
            )

    download_sql = None
    if download_max_rows is not None:
        download_parsed = _enforce_download_limit(parsed.copy(), download_max_rows)
        download_sql = download_parsed.sql(dialect=DIALECT)

    parsed = _enforce_limit(parsed, max_rows)
    sanitized = parsed.sql(dialect=DIALECT)
    if download_sql is None:
        download_sql = sanitized

    return ValidationResult(
        is_valid=True,
        errors=[],
        sanitized_sql=sanitized,
        download_sql=download_sql,
        tables_referenced=referenced_tables,
        repairs=repairs,
    )


def _contains_keyword(upper_sql: str, keyword: str) -> bool:
    """Whole-word (not substring) match, e.g. won't flag CreatedDate as CREATE."""
    import re

    return re.search(rf"\b{keyword}\b", upper_sql) is not None
