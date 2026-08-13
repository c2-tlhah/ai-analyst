from datetime import date

from app.metadata import store
from app.sql.time_context import (
    _period_for_phrase,
    format_time_context_for_prompt,
    question_requires_time_context,
    resolve_relative_time_context,
    validate_relative_time_sql,
)


MIXED_YEAR_SQL = """WITH combined_sales AS (
  SELECT ProductKey, OrderQuantity
  FROM FactInternetSales
  WHERE strftime('%Y', OrderDate) = (
    SELECT strftime('%Y', MAX(OrderDate)) FROM FactInternetSales
  )
  UNION ALL
  SELECT ProductKey, OrderQuantity
  FROM FactResellerSales
  WHERE strftime('%Y', OrderDate) = (
    SELECT strftime('%Y', MAX(OrderDate)) FROM FactResellerSales
  )
)
SELECT ProductKey, SUM(OrderQuantity) FROM combined_sales GROUP BY ProductKey"""


CORRECT_SHARED_YEAR_SQL = """WITH combined_sales AS (
  SELECT ProductKey, OrderQuantity
  FROM FactInternetSales
  WHERE date(OrderDate) >= '2013-01-01' AND date(OrderDate) < '2014-01-01'
  UNION ALL
  SELECT ProductKey, OrderQuantity
  FROM FactResellerSales
  WHERE date(OrderDate) >= '2013-01-01' AND date(OrderDate) < '2014-01-01'
)
SELECT ProductKey, SUM(OrderQuantity) FROM combined_sales GROUP BY ProductKey"""


def _context():
    metadata = store.load_schema_metadata()
    assert metadata is not None
    return resolve_relative_time_context(
        "Which product sold most last year?",
        metadata,
        ["DimProduct", "FactInternetSales", "FactResellerSales"],
        today=date(2026, 8, 13),
    )


def test_historical_last_year_resolves_to_one_shared_calendar_period():
    context = _context()

    assert context["applied"] is True
    assert context["target_year"] == 2013
    assert context["latest_observed_date"] == "2014-01-28"
    prompt = format_time_context_for_prompt(context)
    assert "2013-01-01" in prompt
    assert "2014-01-01" in prompt
    assert "Do not calculate MAX(date)" in prompt


def test_explicit_year_resolves_to_exact_calendar_boundaries_and_coverage():
    metadata = store.load_schema_metadata()
    assert metadata is not None
    context = resolve_relative_time_context(
        "What are the most sold products by month in 2014?",
        metadata,
        ["FactInternetSales", "FactResellerSales"],
        today=date(2026, 8, 13),
    )

    assert context["applied"] is True
    assert context["target_year"] == 2014
    assert context["start_date"] == "2014-01-01"
    assert context["end_date_exclusive"] == "2015-01-01"
    assert context["anchor_policy"] == "explicit_calendar_year"
    assert context["period_coverage"]["FactInternetSales"]["row_count"] == 1970
    assert context["period_coverage"]["FactResellerSales"]["row_count"] == 0
    assert "Results reflect only these available records" in context["coverage_note"]


def test_semantic_guard_rejects_per_table_latest_years():
    errors = validate_relative_time_sql(MIXED_YEAR_SQL, _context())

    assert errors
    assert any("shared resolved range" in error for error in errors)
    assert any("separate MAX date" in error for error in errors)


def test_semantic_guard_accepts_the_shared_resolved_period():
    assert validate_relative_time_sql(CORRECT_SHARED_YEAR_SQL, _context()) == []


def test_semantic_guard_rejects_range_applied_to_only_one_channel():
    one_sided = """WITH combined_sales AS (
      SELECT ProductKey, OrderQuantity FROM FactInternetSales
      WHERE date(OrderDate) >= '2013-01-01' AND date(OrderDate) < '2014-01-01'
      UNION ALL
      SELECT ProductKey, OrderQuantity FROM FactResellerSales
    )
    SELECT ProductKey, SUM(OrderQuantity) FROM combined_sales GROUP BY ProductKey"""

    errors = validate_relative_time_sql(one_sided, _context())

    assert any("every relevant event-source branch" in error for error in errors)


def test_last_month_is_resolved_without_database_specific_names():
    period = _period_for_phrase(
        "show activity last month",
        date(2026, 8, 13),
        date(2026, 8, 12),
    )
    assert period["start_date"] == "2026-07-01"
    assert period["end_date_exclusive"] == "2026-08-01"


def test_explicit_month_quarter_and_year_range_have_exact_boundaries():
    current = date(2026, 8, 13)
    latest = date(2026, 8, 12)

    month = _period_for_phrase("sales in February 2024", current, latest)
    quarter = _period_for_phrase("sales in Q3 2024", current, latest)
    years = _period_for_phrase("sales from 2022 through 2024", current, latest)

    assert (month["start_date"], month["end_date_exclusive"]) == (
        "2024-02-01",
        "2024-03-01",
    )
    assert (quarter["start_date"], quarter["end_date_exclusive"]) == (
        "2024-07-01",
        "2024-10-01",
    )
    assert (years["start_date"], years["end_date_exclusive"]) == (
        "2022-01-01",
        "2025-01-01",
    )


def test_rolling_months_use_calendar_arithmetic_not_thirty_day_approximation():
    period = _period_for_phrase(
        "show activity for the past 12 months",
        date(2024, 3, 31),
        date(2024, 3, 31),
    )

    assert period["start_date"] == "2023-04-01"
    assert period["end_date_exclusive"] == "2024-04-01"


def test_iso_date_does_not_get_misread_as_a_whole_calendar_year():
    assert not question_requires_time_context("sales since 2014-01-15")


def test_identifier_value_that_looks_like_a_year_is_not_a_time_period():
    assert not question_requires_time_context("Show ProductKey 2014")
    assert question_requires_time_context("Show product sales in 2014")


def test_shared_period_supports_different_date_column_names_per_source():
    context = {
        "applied": True,
        "requested": True,
        "start_date": "2026-01-01",
        "end_date_exclusive": "2026-02-01",
        "table_date_columns": {
            "events_a": {"column": "occurred_at"},
            "events_b": {"column": "captured_on"},
        },
    }
    sql = """SELECT occurred_at FROM events_a
      WHERE date(occurred_at) >= '2026-01-01' AND date(occurred_at) < '2026-02-01'
      UNION ALL
      SELECT captured_on FROM events_b
      WHERE date(captured_on) >= '2026-01-01' AND date(captured_on) < '2026-02-01'"""
    assert validate_relative_time_sql(sql, context) == []


def test_shared_period_requires_each_alias_to_have_its_own_filter():
    context = {
        "applied": True,
        "requested": True,
        "start_date": "2026-01-01",
        "end_date_exclusive": "2026-02-01",
        "table_date_columns": {
            "events_a": {"column": "occurred_at"},
            "events_b": {"column": "occurred_at"},
        },
    }
    one_sided = """SELECT a.occurred_at FROM events_a a
      JOIN events_b b ON b.event_id = a.event_id
      WHERE a.occurred_at >= '2026-01-01'
        AND a.occurred_at < '2026-02-01'"""
    both_sides = one_sided + """
        AND b.occurred_at >= '2026-01-01'
        AND b.occurred_at < '2026-02-01'"""

    errors = validate_relative_time_sql(one_sided, context)

    assert any("events_b" in error for error in errors)
    assert validate_relative_time_sql(both_sides, context) == []


def test_aliased_numeric_date_key_normalization_is_recognized_structurally():
    context = {
        "applied": True,
        "requested": True,
        "start_date": "2026-01-01",
        "end_date_exclusive": "2026-02-01",
        "table_date_columns": {
            "events": {
                "column": "DateKey",
                "requires_normalization": True,
            }
        },
    }
    sql = """SELECT e.DateKey FROM events e
      WHERE date(substr(CAST(e.DateKey AS TEXT), 1, 4) || '-' ||
                 substr(CAST(e.DateKey AS TEXT), 5, 2) || '-' ||
                 substr(CAST(e.DateKey AS TEXT), 7, 2)) >= '2026-01-01'
        AND date(substr(CAST(e.DateKey AS TEXT), 1, 4) || '-' ||
                 substr(CAST(e.DateKey AS TEXT), 5, 2) || '-' ||
                 substr(CAST(e.DateKey AS TEXT), 7, 2)) < '2026-02-01'"""

    assert validate_relative_time_sql(sql, context) == []
