"""Deterministic resolution and validation of relative time phrases.

Language models are good at composing joins, but should not independently
decide what a relative period means for each event table. This module resolves one
shared period from the active database before SQL generation and verifies that
the generated SQL uses that period consistently.
"""

from __future__ import annotations

import re
import calendar
from datetime import date, timedelta
from typing import Any

import sqlglot
from sqlglot import exp

from app.db.connection import readonly_connection


_RELATIVE_TIME_RE = re.compile(
    r"\b(?:(?P<modifier>last|past|previous|prior|this|current)\s+"
    r"(?:(?P<count>\d+)\s+)?(?:calendar\s+)?"
    r"(?P<unit>days?|weeks?|months?|years?)|(?P<single>today|yesterday))\b",
    flags=re.IGNORECASE,
)
_YEAR_TOKEN = r"(?:19|20)\d{2}"
_YEAR_RANGE_RE = re.compile(
    rf"\b(?:(?:from\s+)?(?P<start_a>{_YEAR_TOKEN})\s*"
    rf"(?:to|through|until|-)\s*(?P<end_a>{_YEAR_TOKEN})|"
    rf"between\s+(?P<start_b>{_YEAR_TOKEN})\s+and\s+(?P<end_b>{_YEAR_TOKEN}))\b",
    flags=re.IGNORECASE,
)
_QUARTER_YEAR_RE = re.compile(
    rf"\b(?:q(?P<quarter_a>[1-4])\s*(?:of\s+)?(?P<year_a>{_YEAR_TOKEN})|"
    rf"(?P<year_b>{_YEAR_TOKEN})\s*q(?P<quarter_b>[1-4]))\b",
    flags=re.IGNORECASE,
)
_MONTH_NAMES = {
    name.casefold(): index
    for index, name in enumerate(calendar.month_name)
    if name
} | {
    name.casefold(): index
    for index, name in enumerate(calendar.month_abbr)
    if name
}
_MONTH_NAME_PATTERN = "|".join(
    sorted((re.escape(name) for name in _MONTH_NAMES), key=len, reverse=True)
)
_MONTH_YEAR_RE = re.compile(
    rf"\b(?:(?P<month_name>{_MONTH_NAME_PATTERN})\s+(?P<name_year>{_YEAR_TOKEN})|"
    rf"(?P<numeric_year>{_YEAR_TOKEN})-(?P<numeric_month>0?[1-9]|1[0-2])(?!-\d{{2}}))\b",
    flags=re.IGNORECASE,
)
_EXPLICIT_YEAR_RE = re.compile(
    rf"(?<![\d-])(?P<year>{_YEAR_TOKEN})(?![-\d])"
)
_NON_TEMPORAL_YEAR_PREFIXES = frozenset(
    {"id", "key", "code", "sku", "model", "product", "item", "top", "bottom", "first", "limit", "number"}
)
_PREFERRED_EVENT_DATES = (
    "orderdate",
    "transactiondate",
    "saledate",
    "invoicedate",
    "purchasedate",
    "eventdate",
    "createddate",
    "createdat",
    "date",
)
_SECONDARY_DATE_WORDS = frozenset(
    {"due", "ship", "delivery", "end", "start", "birth", "updated", "modified"}
)


def _explicit_year_match(question: str) -> re.Match[str] | None:
    for match in _EXPLICIT_YEAR_RE.finditer(question or ""):
        prefix = (question or "")[: match.start()]
        humanized_prefix = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", prefix)
        prefix_words = re.findall(r"[a-z0-9]+", humanized_prefix.casefold())
        if prefix_words and prefix_words[-1] in _NON_TEMPORAL_YEAR_PREFIXES:
            continue
        return match
    return None


def question_requires_time_context(question: str) -> bool:
    """Whether a question contains a supported relative or explicit period."""
    value = question or ""
    return bool(
        _RELATIVE_TIME_RE.search(value)
        or _YEAR_RANGE_RE.search(value)
        or _QUARTER_YEAR_RE.search(value)
        or _MONTH_YEAR_RE.search(value)
        or _explicit_year_match(value)
    )


def _add_months(value: date, months: int) -> date:
    absolute = value.year * 12 + value.month - 1 + months
    return date(absolute // 12, absolute % 12 + 1, 1)


def _shift_months(value: date, months: int) -> date:
    """Shift a date by calendar months while clamping its day if necessary."""
    absolute = value.year * 12 + value.month - 1 + months
    year, month = absolute // 12, absolute % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _period_for_phrase(question: str, current: date, latest: date) -> dict[str, Any]:
    year_range = _YEAR_RANGE_RE.search(question)
    if year_range:
        start_year = int(year_range.group("start_a") or year_range.group("start_b"))
        end_year = int(year_range.group("end_a") or year_range.group("end_b"))
        start_year, end_year = min(start_year, end_year), max(start_year, end_year)
        return {
            "phrase": year_range.group(0).casefold(),
            "period_kind": "inclusive_calendar_year_range",
            "start_date": date(start_year, 1, 1).isoformat(),
            "end_date_exclusive": date(end_year + 1, 1, 1).isoformat(),
            "target_year": None,
            "target_years": [start_year, end_year],
            "anchor_policy": "explicit_calendar_year_range",
        }
    quarter_year = _QUARTER_YEAR_RE.search(question)
    if quarter_year:
        quarter = int(quarter_year.group("quarter_a") or quarter_year.group("quarter_b"))
        year = int(quarter_year.group("year_a") or quarter_year.group("year_b"))
        start_month = (quarter - 1) * 3 + 1
        start = date(year, start_month, 1)
        end = _add_months(start, 3)
        return {
            "phrase": quarter_year.group(0).casefold(),
            "period_kind": "calendar_quarter",
            "start_date": start.isoformat(),
            "end_date_exclusive": end.isoformat(),
            "target_year": year,
            "target_quarter": quarter,
            "anchor_policy": "explicit_calendar_quarter",
        }
    month_year = _MONTH_YEAR_RE.search(question)
    if month_year:
        if month_year.group("month_name"):
            month = _MONTH_NAMES[month_year.group("month_name").casefold()]
            year = int(month_year.group("name_year"))
        else:
            month = int(month_year.group("numeric_month"))
            year = int(month_year.group("numeric_year"))
        start = date(year, month, 1)
        end = _add_months(start, 1)
        return {
            "phrase": month_year.group(0).casefold(),
            "period_kind": "calendar_month",
            "start_date": start.isoformat(),
            "end_date_exclusive": end.isoformat(),
            "target_year": year,
            "target_month": month,
            "anchor_policy": "explicit_calendar_month",
        }
    explicit_year = _explicit_year_match(question)
    if explicit_year:
        year = int(explicit_year.group("year"))
        return {
            "phrase": explicit_year.group(0).casefold(),
            "period_kind": "calendar_year",
            "start_date": date(year, 1, 1).isoformat(),
            "end_date_exclusive": date(year + 1, 1, 1).isoformat(),
            "target_year": year,
            "anchor_policy": "explicit_calendar_year",
        }
    match = _RELATIVE_TIME_RE.search(question)
    if not match:  # guarded by question_requires_time_context
        return {}
    phrase = match.group(0).casefold()
    single = match.group("single")
    modifier = (match.group("modifier") or "").casefold()
    unit = (match.group("unit") or "day").casefold().rstrip("s")
    count = int(match.group("count") or 1)

    if unit == "year":
        current_enough = latest.year >= current.year - 1
    elif unit == "month":
        current_enough = latest >= _add_months(current.replace(day=1), -1)
    elif unit == "week":
        current_enough = latest >= current - timedelta(days=current.weekday() + 7)
    else:
        current_enough = latest >= current - timedelta(days=max(1, count))
    anchor = current if current_enough else latest

    if single == "today":
        start, end, kind = anchor, anchor + timedelta(days=1), "calendar_day"
    elif single == "yesterday":
        start, end, kind = anchor - timedelta(days=1), anchor, "calendar_day"
    elif modifier in {"this", "current"}:
        if unit == "year":
            start, end = date(anchor.year, 1, 1), date(anchor.year + 1, 1, 1)
        elif unit == "month":
            start, end = anchor.replace(day=1), _add_months(anchor, 1)
        elif unit == "week":
            start = anchor - timedelta(days=anchor.weekday())
            end = start + timedelta(days=7)
        else:
            start, end = anchor, anchor + timedelta(days=1)
        kind = f"calendar_{unit}"
    elif count == 1 and modifier in {"last", "previous", "prior"}:
        if unit == "year":
            start, end = date(anchor.year - 1, 1, 1), date(anchor.year, 1, 1)
        elif unit == "month":
            end = anchor.replace(day=1)
            start = _add_months(end, -1)
        elif unit == "week":
            end = anchor - timedelta(days=anchor.weekday())
            start = end - timedelta(days=7)
        else:
            start, end = anchor - timedelta(days=1), anchor
        kind = f"previous_calendar_{unit}"
    else:
        # Explicit plural windows are inclusive of the anchor date and use one
        # exclusive end, making all table filters identical and timezone-free.
        end = anchor + timedelta(days=1)
        if unit == "year":
            start = _shift_months(end, -12 * count)
        elif unit == "month":
            start = _shift_months(end, -count)
        else:
            days = count * {"day": 1, "week": 7}[unit]
            start = anchor - timedelta(days=days - 1)
        kind = f"rolling_{count}_{unit}s"

    return {
        "phrase": phrase,
        "period_kind": kind,
        "start_date": start.isoformat(),
        "end_date_exclusive": end.isoformat(),
        "target_year": start.year if unit == "year" and count == 1 else None,
        "anchor_policy": (
            "current_calendar_date" if current_enough else "historical_database_latest_date"
        ),
    }


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _column_score(name: str, sql_type: str) -> int:
    compact = re.sub(r"[^a-z0-9]", "", name.casefold())
    words = set(re.findall(r"[a-z]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)))
    score = 0
    if compact in _PREFERRED_EVENT_DATES:
        score += 100 - _PREFERRED_EVENT_DATES.index(compact)
    if "order" in compact or "transaction" in compact or "sale" in compact:
        score += 35
    if compact.endswith("date") or compact.endswith("timestamp"):
        score += 20
    if compact.endswith("datekey") or compact.endswith("timekey"):
        score -= 15
    if words & _SECONDARY_DATE_WORDS or any(word in compact for word in _SECONDARY_DATE_WORDS):
        score -= 60
    if any(token in (sql_type or "").upper() for token in ("DATE", "TIME", "TEXT", "CHAR")):
        score += 10
    return score


def _date_expression(column: str, sql_type: str) -> str:
    quoted = _quote_identifier(column)
    compact = re.sub(r"[^a-z0-9]", "", column.casefold())
    numeric = any(
        token in (sql_type or "").upper()
        for token in ("INT", "NUM", "DEC", "REAL")
    )
    if numeric and compact.endswith(("datekey", "timekey")):
        text = f"CAST({quoted} AS TEXT)"
        return (
            f"date(substr({text}, 1, 4) || '-' || substr({text}, 5, 2) || "
            f"'-' || substr({text}, 7, 2))"
        )
    return f"date({quoted})"


def _candidate_event_columns(
    metadata: dict[str, Any], table_names: list[str], question: str
) -> dict[str, dict[str, str]]:
    tables = metadata.get("tables", {})
    candidates: dict[str, dict[str, str]] = {}
    for table_name in table_names:
        table = tables.get(table_name) or {}
        ranked = []
        for column_name, column in (table.get("columns") or {}).items():
            if column.get("semantic_role") != "temporal":
                continue
            sql_type = str(column.get("sql_type") or "")
            ranked.append((_column_score(column_name, sql_type), column_name, sql_type))
        if ranked:
            _score, column_name, sql_type = max(ranked)
            candidates[table_name] = {
                "column": column_name,
                "expression": _date_expression(column_name, sql_type),
                "requires_normalization": bool(
                    any(
                        token in sql_type.upper()
                        for token in ("INT", "NUM", "DEC", "REAL")
                    )
                    and re.sub(r"[^a-z0-9]", "", column_name.casefold()).endswith(
                        ("datekey", "timekey")
                    )
                ),
                "table_kind": str(table.get("kind") or "unknown"),
            }
    event_sources = {
        name: candidate
        for name, candidate in candidates.items()
        if candidate["table_kind"] != "dimension"
    }
    explicitly_named_dimensions = {
        name: candidate
        for name, candidate in candidates.items()
        if candidate["table_kind"] == "dimension"
        and re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
            question,
            flags=re.IGNORECASE,
        )
    }
    selected = event_sources or candidates
    return {**selected, **explicitly_named_dimensions}


def resolve_relative_time_context(
    question: str,
    metadata: dict[str, Any],
    table_names: list[str],
    *,
    today: date | None = None,
) -> dict[str, Any]:
    """Resolve one shared date range for relative wording such as "last year".

    For a current database, last year follows the wall clock. For a historical
    database more than one year behind the wall clock, it means the calendar
    year preceding the database's latest observed event date. This makes stale
    demo/warehouse snapshots useful without silently mixing table-specific years.
    """
    if not question_requires_time_context(question):
        return {"applied": False}

    candidates = _candidate_event_columns(metadata, table_names, question)
    coverage: dict[str, dict[str, str | None]] = {}
    observed_dates: list[date] = []
    with readonly_connection() as connection:
        for table_name, candidate in candidates.items():
            expression = candidate["expression"]
            row = connection.execute(
                f"SELECT MIN({expression}), MAX({expression}) "
                f"FROM {_quote_identifier(table_name)} "
                f"WHERE {expression} IS NOT NULL"
            ).fetchone()
            minimum = str(row[0]) if row and row[0] else None
            maximum = str(row[1]) if row and row[1] else None
            coverage[table_name] = {
                "column": candidate["column"],
                "filter_expression": candidate["expression"],
                "requires_normalization": candidate["requires_normalization"],
                "minimum": minimum,
                "maximum": maximum,
            }
            if maximum:
                try:
                    observed_dates.append(date.fromisoformat(maximum[:10]))
                except ValueError:
                    continue

    explicit_period = bool(
        _YEAR_RANGE_RE.search(question)
        or _QUARTER_YEAR_RE.search(question)
        or _MONTH_YEAR_RE.search(question)
        or _explicit_year_match(question)
    )
    if not candidates:
        return {
            "applied": False,
            "requested": True,
            "reason": "No usable event-date field was found in the relevant tables.",
        }
    if not observed_dates and not explicit_period:
        return {
            "applied": False,
            "requested": True,
            "reason": "No usable event-date coverage was found in the relevant tables.",
        }

    current = today or date.today()
    latest = max(observed_dates) if observed_dates else current
    # If data reaches the current or immediately previous calendar year, use
    # the ordinary wall-clock interpretation. Otherwise anchor to the historic
    # snapshot so a question does not return an empty modern year.
    period = _period_for_phrase(question, current, latest)
    start_date = period["start_date"]
    end_date = period["end_date_exclusive"]
    period_coverage: dict[str, dict[str, Any]] = {}
    with readonly_connection() as connection:
        for table_name, candidate in candidates.items():
            expression = candidate["expression"]
            row = connection.execute(
                f"SELECT MIN({expression}), MAX({expression}), COUNT(*) "
                f"FROM {_quote_identifier(table_name)} "
                f"WHERE {expression} >= ? AND {expression} < ?",
                (start_date, end_date),
            ).fetchone()
            period_coverage[table_name] = {
                "row_count": int(row[2] or 0) if row else 0,
                "minimum": str(row[0]) if row and row[0] else None,
                "maximum": str(row[1]) if row and row[1] else None,
            }
    coverage_parts = []
    for table_name, details in period_coverage.items():
        if details["row_count"]:
            coverage_parts.append(
                f"{table_name}: {details['row_count']:,} row(s), "
                f"{details['minimum']} to {details['maximum']}"
            )
        else:
            coverage_parts.append(f"{table_name}: 0 rows")
    return {
        "applied": True,
        "requested": True,
        **period,
        "start_date": start_date,
        "end_date_exclusive": end_date,
        "latest_observed_date": max(observed_dates).isoformat() if observed_dates else None,
        "label": (
            f"{period['phrase'].title()} resolved to "
            f"({start_date} inclusive to {end_date} exclusive)."
        ),
        "table_date_columns": coverage,
        "period_coverage": period_coverage,
        "coverage_note": (
            "Available source records in this period — "
            + "; ".join(coverage_parts)
            + ". Results reflect only these available records."
            if coverage_parts
            else ""
        ),
    }


def format_time_context_for_prompt(context: dict[str, Any]) -> str:
    if not context.get("applied"):
        if context.get("requested"):
            return (
                "RESOLVED TIME CONTEXT: unavailable. "
                f"{context.get('reason', 'No usable temporal field was found.')} "
                "Do not invent a date field or silently reinterpret the period."
            )
        return ""
    fields = ", ".join(
        f"{table}: {details.get('filter_expression') or details['column']}"
        for table, details in context.get("table_date_columns", {}).items()
    )
    return (
        "RESOLVED TIME CONTEXT (deterministic and mandatory):\n"
        f"- The phrase {context['phrase']!r} means {context['period_kind']}.\n"
        f"- Apply one shared range to every relevant event source: date >= "
        f"'{context['start_date']}' AND date < '{context['end_date_exclusive']}'.\n"
        f"- Event-date columns: {fields or 'none resolved'}.\n"
        "- Do not calculate MAX(date), latest year, or separate date anchors per table.\n"
        "- Use the same literal start and exclusive-end boundaries in every UNION branch."
    )


def validate_relative_time_sql(sql: str, context: dict[str, Any]) -> list[str]:
    """Reject a query that ignores or changes the deterministic shared period."""
    if context.get("requested") and not context.get("applied"):
        return [
                "The question uses a time period, but the relevant live schema "
            "has no usable event-date field. Name an exact date range or connect data "
            "with a documented temporal column."
        ]
    if not context.get("applied"):
        return []
    errors: list[str] = []
    start = str(context.get("start_date") or "")
    end = str(context.get("end_date_exclusive") or "")
    if start not in sql or end not in sql:
        errors.append(
            "Time-period mismatch: use the shared resolved range "
            f">= '{start}' and < '{end}' for every relevant source."
        )
    if re.search(r"\bMAX\s*\([^)]*(?:date|time)", sql, flags=re.IGNORECASE):
        errors.append(
            "Do not derive a separate MAX date/year inside SQL; use the resolved shared period."
        )

    try:
        parsed = sqlglot.parse_one(sql, dialect="sqlite")
    except Exception:  # syntax errors are reported by the primary validator
        return list(dict.fromkeys(errors))
    coverage_by_lower = {
        table.casefold(): (table, details)
        for table, details in context.get("table_date_columns", {}).items()
    }
    referenced_sources = 0
    for table_node in parsed.find_all(exp.Table):
        coverage = coverage_by_lower.get(table_node.name.casefold())
        if coverage is None:
            continue
        referenced_sources += 1
        table, details = coverage
        select = table_node.find_ancestor(exp.Select)
        column = str(details.get("column") or "")
        alias = table_node.alias_or_name.casefold()
        predicate_roots: list[exp.Expression] = []
        if select is not None and select.args.get("where") is not None:
            predicate_roots.append(select.args["where"].this)
        if select is not None:
            predicate_roots.extend(
                join.args["on"]
                for join in select.args.get("joins") or []
                if join.args.get("on") is not None
            )
        direct_tables = [
            node
            for node in (select.find_all(exp.Table) if select is not None else [])
            if node.find_ancestor(exp.Select) is select
        ]
        unqualified_is_ambiguous = sum(
            1
            for node in direct_tables
            if node.name.casefold() in coverage_by_lower
        ) > 1
        found_start = False
        found_end = False
        normalized_target = not details.get("requires_normalization")
        comparison_types = (exp.EQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between)
        for root in predicate_roots:
            comparisons = [root] if isinstance(root, comparison_types) else []
            comparisons.extend(root.find_all(*comparison_types))
            for comparison in comparisons:
                target_columns = [
                    candidate
                    for candidate in comparison.find_all(exp.Column)
                    if candidate.name.casefold() == column.casefold()
                    and (
                        candidate.table.casefold() == alias
                        if candidate.table
                        else not unqualified_is_ambiguous
                    )
                ]
                if not target_columns:
                    continue
                comparison_sql = comparison.sql(dialect="sqlite").casefold()
                found_start = found_start or start.casefold() in comparison_sql
                found_end = found_end or end.casefold() in comparison_sql
                if (
                    details.get("requires_normalization")
                    and "date(" in comparison_sql
                    and ("substr(" in comparison_sql or "substring(" in comparison_sql)
                    and "cast(" in comparison_sql
                ):
                    normalized_target = True
        source_label = (
            f"{table} (alias {table_node.alias_or_name})"
            if table_node.alias
            else table
        )
        if not found_start or not found_end:
            errors.append(
                "Apply both shared date boundaries inside every relevant event-source "
                f"branch; they are missing for {source_label}."
            )
        if not normalized_target:
            errors.append(
                f"The numeric date key {source_label}.{column} must be normalized "
                "with date(substr(CAST(...))) before comparison."
            )
    if referenced_sources == 0:
        errors.append(
            "The relative-time query does not reference a source with a resolved event date."
        )
    return errors
