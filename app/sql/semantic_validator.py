"""Database-agnostic semantic checks for model-generated SQLite.

The security validator proves that a statement is read-only.  This module uses
the active database's discovered capabilities to reject a second class of bad
queries: valid SQL that references unknown/ambiguous columns, joins tables on an
undocumented path, aggregates labels as numbers, or violates requested ranking
semantics.
"""

from __future__ import annotations

import re
from typing import Any

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify

from app.sql.repair import question_group_words


def _schema(metadata: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        table: {
            column: str(details.get("sql_type") or "TEXT")
            for column, details in info.get("columns", {}).items()
        }
        for table, info in metadata.get("tables", {}).items()
    }


def _table_aliases(expression: exp.Expression, metadata: dict[str, Any]) -> dict[str, str]:
    known = {name.casefold(): name for name in metadata.get("tables", {})}
    aliases: dict[str, str] = {}
    for table in expression.find_all(exp.Table):
        canonical = known.get(table.name.casefold())
        if canonical:
            aliases[table.alias_or_name.casefold()] = canonical
            aliases.setdefault(table.name.casefold(), canonical)
    return aliases


def _select_table_aliases(
    select: exp.Select | None, metadata: dict[str, Any]
) -> dict[str, str]:
    """Resolve real-table aliases only within one SELECT scope.

    Alias names may be reused in separate CTEs or UNION branches. A module-wide
    alias map can therefore assign a column to the wrong table and either reject
    valid SQL or miss a type error.
    """
    if select is None:
        return {}
    known = {name.casefold(): name for name in metadata.get("tables", {})}
    aliases: dict[str, str] = {}
    for table in select.find_all(exp.Table):
        if table.find_ancestor(exp.Select) is not select:
            continue
        canonical = known.get(table.name.casefold())
        if canonical:
            aliases[table.alias_or_name.casefold()] = canonical
            aliases.setdefault(table.name.casefold(), canonical)
    return aliases


def _column_details(
    column: exp.Column,
    aliases: dict[str, str],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    tables = metadata.get("tables", {})
    if column.table:
        table = aliases.get(column.table.casefold())
        if table:
            columns = tables.get(table, {}).get("columns", {})
            canonical = {name.casefold(): value for name, value in columns.items()}
            return canonical.get(column.name.casefold())
        return None
    matches = [
        details
        for table in set(aliases.values())
        if (
            details := {
                name.casefold(): value
                for name, value in tables.get(table, {}).get("columns", {}).items()
            }.get(column.name.casefold())
        )
    ]
    return matches[0] if len(matches) == 1 else None


def _is_numeric(details: dict[str, Any] | None) -> bool:
    if not details:
        return True  # Derived/CTE output; its source columns are checked independently.
    if details.get("semantic_role") in {"key", "temporal", "categorical_attribute"}:
        return False
    observed_family = details.get("observed_value_family")
    if observed_family in {"text", "temporal_text", "mixed", "blob"}:
        return False
    if observed_family in {"numeric", "numeric_text"}:
        return True
    if details.get("semantic_role") in {"measure", "numeric_attribute"}:
        return True
    if details.get("declared_type_family") == "numeric":
        return True
    observed = set(details.get("observed_storage_types") or [])
    return bool(observed) and observed <= {"integer", "real"}


def _is_temporal(details: dict[str, Any] | None, column_name: str = "") -> bool:
    if not details:
        return True
    observed_family = details.get("observed_value_family")
    compact_name = re.sub(r"[^a-z0-9]", "", column_name.casefold())
    if observed_family in {"numeric", "numeric_text"} and compact_name.endswith(
        ("datekey", "timekey")
    ):
        return True
    if observed_family in {"numeric", "numeric_text", "mixed", "blob"}:
        return False
    if observed_family == "text" and details.get("sampled_non_null_count", 0):
        return False
    return (
        details.get("semantic_role") == "temporal"
        or details.get("declared_type_family") == "temporal"
        or details.get("observed_value_family") == "temporal_text"
    )


def _relationship_groups(
    metadata: dict[str, Any],
) -> list[set[frozenset[tuple[str, str]]]]:
    grouped: dict[tuple[Any, ...], set[frozenset[tuple[str, str]]]] = {}
    for index, rel in enumerate(metadata.get("relationships", [])):
        constraint_id = rel.get("constraint_id")
        constraint_size = int(rel.get("constraint_size") or 1)
        key = (
            rel["from_table"].casefold(),
            rel["to_table"].casefold(),
            constraint_id,
        ) if constraint_size > 1 and constraint_id is not None else ("single", index)
        grouped.setdefault(key, set()).add(
            frozenset(
                {
                    (rel["from_table"].casefold(), rel["from_column"].casefold()),
                    (rel["to_table"].casefold(), rel["to_column"].casefold()),
                }
            )
        )
    return list(grouped.values())


def _validate_joins(
    expression: exp.Expression,
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    valid_groups = _relationship_groups(metadata)
    for select in expression.find_all(exp.Select):
        aliases = _select_table_aliases(select, metadata)
        for join in select.args.get("joins") or []:
            right = join.this
            if not isinstance(right, exp.Table):
                continue
            right_table = aliases.get(right.alias_or_name.casefold())
            # CTEs and derived sources do not have catalog relationships of their
            # own. Their underlying real-table joins are checked in their scopes.
            if not right_table:
                continue
            on = join.args.get("on")
            if on is None:
                errors.append(
                    "Every direct real-table join must use an explicit documented "
                    "key relationship."
                )
                continue
            equality_pairs: set[frozenset[tuple[str, str]]] = set()
            for equality in on.find_all(exp.EQ):
                left_col, right_col = equality.left, equality.right
                if not isinstance(left_col, exp.Column) or not isinstance(
                    right_col, exp.Column
                ):
                    continue
                left_table = (
                    aliases.get(left_col.table.casefold()) if left_col.table else None
                )
                other_table = (
                    aliases.get(right_col.table.casefold()) if right_col.table else None
                )
                if left_table and other_table:
                    equality_pairs.add(
                        frozenset(
                            {
                                (left_table.casefold(), left_col.name.casefold()),
                                (other_table.casefold(), right_col.name.casefold()),
                            }
                        )
                    )
            on_aliases = {
                column.table.casefold()
                for column in on.find_all(exp.Column)
                if column.table
            }
            if any(alias not in aliases for alias in on_aliases):
                continue
            if not any(group <= equality_pairs for group in valid_groups):
                errors.append(
                    f"Join to '{right_table}' does not use a declared or verified "
                    "inferred relationship from the retrieved schema."
                )
    return errors


def _validate_aggregate_types(
    expression: exp.Expression,
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for aggregate in expression.find_all(exp.Sum, exp.Avg):
        aliases = _select_table_aliases(aggregate.find_ancestor(exp.Select), metadata)
        for column in aggregate.find_all(exp.Column):
            details = _column_details(column, aliases, metadata)
            if not _is_numeric(details):
                errors.append(
                    f"{aggregate.key.upper()} requires numeric data, but '{column}' is "
                    "documented/profiled as non-numeric."
                )
    date_function_keys = {
        "date", "datetime", "date_trunc", "time_to_str", "str_to_time",
        "ts_or_ds_to_date", "unix_to_time",
    }
    for function in expression.walk():
        if function.key not in date_function_keys:
            continue
        aliases = _select_table_aliases(function.find_ancestor(exp.Select), metadata)
        for column in function.find_all(exp.Column):
            details = _column_details(column, aliases, metadata)
            if not _is_temporal(details, column.name):
                errors.append(
                    f"Date/time function uses '{column}', which is not a discovered "
                    "temporal field."
                )
    return errors


def _validate_fact_fanout(
    expression: exp.Expression, metadata: dict[str, Any]
) -> list[str]:
    """Reject aggregate scopes that join multiple raw fact/event tables."""
    errors: list[str] = []
    tables = metadata.get("tables", {})
    for select in expression.find_all(exp.Select):
        direct_aggregates = [
            aggregate
            for aggregate in select.find_all(exp.AggFunc)
            if aggregate.find_ancestor(exp.Select) is select
            and not aggregate.find_ancestor(exp.Window)
        ]
        if not direct_aggregates:
            continue
        aliases = _select_table_aliases(select, metadata)
        fact_tables = sorted(
            {
                table
                for table in aliases.values()
                if str(tables.get(table, {}).get("kind") or "").casefold()
                in {"fact", "event"}
            }
        )
        if len(fact_tables) > 1:
            errors.append(
                "This aggregate joins multiple raw fact/event tables in one query "
                f"scope ({', '.join(fact_tables)}), which can multiply rows and "
                "overstate measures. Aggregate each source to the requested grain "
                "in separate CTEs, then combine those aggregates with UNION ALL or "
                "a grain-preserving join."
            )
    return errors


def _validate_default_aggregations(
    expression: exp.Expression,
    metadata: dict[str, Any],
    question: str,
) -> list[str]:
    """Enforce profiled aggregation defaults only when wording is ambiguous."""
    normalized = " ".join((question or "").casefold().split())
    explicit_aggregation = bool(re.search(
        r"\b(sum|total|average|avg|mean|count|how many|minimum|min|maximum|max|"
        r"highest|lowest)\b",
        normalized,
    ))
    errors: list[str] = []
    asks_count = bool(re.search(r"\b(count|how many|number of)\b", normalized))
    if not asks_count:
        for aggregate in expression.find_all(exp.Count):
            argument = aggregate.this
            if not isinstance(argument, exp.Column):
                continue
            aliases = _select_table_aliases(
                aggregate.find_ancestor(exp.Select), metadata
            )
            details = _column_details(argument, aliases, metadata)
            expected = str(
                (details or {}).get("default_aggregation") or ""
            ).casefold()
            if (
                (details or {}).get("semantic_role") == "measure"
                and expected in {"sum", "avg"}
            ):
                errors.append(
                    f"COUNT('{argument}') changes the meaning of a documented "
                    f"measure whose default aggregation is {expected.upper()}. "
                    "Use that default aggregation or explicitly ask for a count."
                )
    if explicit_aggregation:
        return errors
    for aggregate in expression.find_all(exp.Sum, exp.Avg):
        argument = aggregate.this
        if not isinstance(argument, exp.Column):
            continue
        aliases = _select_table_aliases(aggregate.find_ancestor(exp.Select), metadata)
        details = _column_details(argument, aliases, metadata)
        expected = str((details or {}).get("default_aggregation") or "").casefold()
        actual = aggregate.key.casefold()
        if expected in {"sum", "avg"} and actual != expected:
            errors.append(
                f"The live metadata defines {expected.upper()} as the default "
                f"aggregation for '{argument}'. The question does not explicitly "
                f"request {actual.upper()}, so use {expected.upper()} or clarify the "
                "aggregation in the question."
            )
    return errors


def _validate_grouping(expression: exp.Expression) -> list[str]:
    errors: list[str] = []
    for select in expression.find_all(exp.Select):
        aggregates = [
            agg
            for agg in select.find_all(exp.AggFunc)
            if agg.find_ancestor(exp.Select) is select and not agg.find_ancestor(exp.Window)
        ]
        if not aggregates:
            continue
        group = select.args.get("group")
        grouped_sql = {
            item.sql(dialect="sqlite").casefold()
            for item in (group.expressions if group else [])
        }
        grouped_names = {
            item.name.casefold()
            for item in (group.expressions if group else [])
            if isinstance(item, exp.Column)
        }
        grouped_ordinals = {
            int(item.this)
            for item in (group.expressions if group else [])
            if isinstance(item, exp.Literal) and not item.is_string
        }
        for projection_index, projection in enumerate(select.expressions, start=1):
            value = projection.this if isinstance(projection, exp.Alias) else projection
            if isinstance(value, exp.Literal) or value.find(exp.AggFunc):
                continue
            value_sql = value.sql(dialect="sqlite").casefold()
            projection_alias = (
                projection.alias.casefold() if isinstance(projection, exp.Alias) else ""
            )
            columns = list(value.find_all(exp.Column))
            if (
                value_sql in grouped_sql
                or projection_alias in grouped_names
                or projection_index in grouped_ordinals
                or all(
                column.name.casefold() in grouped_names for column in columns
                )
            ):
                continue
            if columns:
                errors.append(
                    f"Selected expression '{value.sql(dialect='sqlite')}' is neither "
                    "aggregated nor included in GROUP BY. SQLite would return an "
                    "arbitrary value for it."
                )
    return errors


def _literal_limit(expression: exp.Expression) -> int | None:
    limit = expression.args.get("limit")
    try:
        return int(limit.expression.this) if limit is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def _integer_literal(value: exp.Expression | None) -> int | None:
    if not isinstance(value, exp.Literal) or value.is_string:
        return None
    try:
        return int(value.this)
    except (TypeError, ValueError):
        return None


def _rank_filter_limit(expression: exp.Expression, alias: str) -> int | None:
    """Return the per-partition row cutoff applied to a window-rank alias."""
    alias_lower = alias.casefold()
    for comparison in expression.find_all(exp.EQ, exp.LTE, exp.LT, exp.GTE, exp.GT):
        left, right = comparison.left, comparison.right
        left_is_rank = isinstance(left, exp.Column) and left.name.casefold() == alias_lower
        right_is_rank = isinstance(right, exp.Column) and right.name.casefold() == alias_lower
        left_value = _integer_literal(left)
        right_value = _integer_literal(right)

        if isinstance(comparison, exp.EQ):
            if left_is_rank and right_value is not None:
                return right_value
            if right_is_rank and left_value is not None:
                return left_value
        elif isinstance(comparison, exp.LTE):
            if left_is_rank and right_value is not None:
                return right_value
        elif isinstance(comparison, exp.LT):
            if left_is_rank and right_value is not None:
                return right_value - 1
        elif isinstance(comparison, exp.GTE):
            if right_is_rank and left_value is not None:
                return left_value
        elif isinstance(comparison, exp.GT):
            if right_is_rank and left_value is not None:
                return left_value - 1
    for between in expression.find_all(exp.Between):
        target = between.this
        if not isinstance(target, exp.Column) or target.name.casefold() != alias_lower:
            continue
        low = _integer_literal(between.args.get("low"))
        high = _integer_literal(between.args.get("high"))
        if low == 1 and high is not None:
            return high
    return None


def _window_rankings(expression: exp.Expression) -> list[dict[str, Any]]:
    """Describe ROW_NUMBER/RANK/DENSE_RANK windows used for ranked results."""
    rankings: list[dict[str, Any]] = []
    for window in expression.find_all(exp.Window):
        function_name = re.sub(
            r"[^a-z]", "", str(getattr(window.this, "key", "")).casefold()
        )
        if function_name not in {"rownumber", "rank", "denserank"}:
            continue
        partitions = window.args.get("partition_by") or []
        parent = window.parent
        alias = parent.alias if isinstance(parent, exp.Alias) else ""
        order = window.args.get("order")
        ordered = list(order.expressions) if order is not None else []
        partition_words: set[str] = set()
        for partition in partitions:
            for column in partition.find_all(exp.Column):
                partition_words.update(
                    re.findall(r"[a-z0-9]+", _humanized_identifier(column.name))
                )
        rankings.append(
            {
                "alias": alias,
                "cutoff": _rank_filter_limit(expression, alias) if alias else None,
                "has_order": bool(ordered),
                "descending": bool(ordered[0].args.get("desc")) if ordered else None,
                "partition_words": partition_words,
                "partitioned": bool(partitions),
            }
        )
    return rankings


def _ordered_limit_rankings(expression: exp.Expression) -> list[dict[str, Any]]:
    """Describe ORDER BY ... LIMIT query scopes, including inner CTEs."""
    rankings: list[dict[str, Any]] = []
    for select in expression.find_all(exp.Select):
        order = select.args.get("order")
        ordered = list(order.expressions) if order is not None else []
        cutoff = _literal_limit(select)
        if not ordered or cutoff is None:
            continue
        rankings.append(
            {
                "cutoff": cutoff,
                "descending": bool(ordered[0].args.get("desc")),
            }
        )
    return rankings


def _validate_question_alignment(
    expression: exp.Expression,
    question: str,
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    normalized = " ".join((question or "").casefold().split())
    has_sum = any(True for _ in expression.find_all(exp.Sum))
    has_avg = any(True for _ in expression.find_all(exp.Avg))
    has_count = any(True for _ in expression.find_all(exp.Count))
    has_max = any(True for _ in expression.find_all(exp.Max))
    has_min = any(True for _ in expression.find_all(exp.Min))
    if re.search(r"\b(average|avg|mean)\b", normalized) and not has_avg:
        errors.append("An average/mean question requires an explicit AVG aggregation.")
    asks_count = bool(
        re.search(r"\b(count|how many|number of|number of the)\b", normalized)
    )
    if asks_count and not has_count:
        errors.append("A count/how-many question requires COUNT(...).")
    if re.search(r"\bsum\b", normalized) and not has_sum:
        errors.append("A question explicitly requesting a sum requires SUM(...).")
    if re.search(r"\btotal\b", normalized) and not asks_count:
        numeric_measure_mentioned = any(
            details.get("semantic_role") in {"measure", "numeric_attribute"}
            and _is_numeric(details)
            and _question_mentions_identifier(question, column)
            for table in metadata.get("tables", {}).values()
            for column, details in table.get("columns", {}).items()
        )
        if numeric_measure_mentioned and not has_sum:
            errors.append(
                "The question requests the total of a documented numeric measure, "
                "so SQL must use SUM(...), not a row count."
            )
        elif not numeric_measure_mentioned and not (has_sum or has_count):
            errors.append(
                "A total question requires an explicit SUM or COUNT aggregation."
            )
    asks_unique = bool(re.search(r"\b(unique|distinct)\b", normalized))
    if asks_unique and has_count:
        if not any(count.find(exp.Distinct) for count in expression.find_all(exp.Count)):
            errors.append("A unique/distinct count must use COUNT(DISTINCT ...).")
    elif asks_unique:
        has_distinct_result = any(
            bool(select.args.get("distinct") or select.args.get("group"))
            for select in expression.find_all(exp.Select)
        ) or any(
            union.args.get("distinct") is not False
            for union in expression.find_all(exp.Union)
        )
        if not has_distinct_result:
            errors.append(
                "A unique/distinct result must use SELECT DISTINCT, GROUP BY, "
                "UNION, or COUNT(DISTINCT ...)."
            )

    ranking_text = re.sub(r"\bat\s+(?:least|most)\b", "", normalized)
    ranking_high = bool(
        re.search(
            r"\b(top|most|highest|largest|greatest|best|latest|newest)\b",
            ranking_text,
        )
    )
    ranking_low = bool(
        re.search(
            r"\b(bottom|least|lowest|smallest|worst|earliest|oldest)\b",
            ranking_text,
        )
    )
    top_match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", ranking_text)
    if ranking_high or ranking_low:
        requested = int(top_match.group(1)) if top_match else 1
        explicit_maximum = bool(
            re.search(r"\b(highest|largest|greatest|maximum|max)\b", normalized)
        )
        explicit_minimum = bool(
            re.search(r"\b(lowest|smallest|minimum|min)\b", normalized)
        )
        extrema_satisfied = (
            (not ranking_high or (explicit_maximum and has_max))
            and (not ranking_low or (explicit_minimum and has_min))
            and not top_match
        )
        if extrema_satisfied:
            return errors

        all_windows = _window_rankings(expression)
        all_partitioned = [
            ranking for ranking in all_windows if ranking["partitioned"]
        ]
        partition_words = set().union(
            *(ranking["partition_words"] for ranking in all_partitioned)
        ) if all_partitioned else set()
        grouped_wording = bool(partition_words & question_group_words(normalized))
        partitioned = all_partitioned if grouped_wording else []
        if partitioned:
            correctly_ordered = [
                ranking
                for ranking in partitioned
                if ranking["has_order"]
                and (
                    (ranking_high and ranking["descending"] is True)
                    or (ranking_low and ranking["descending"] is False)
                )
            ]
            if not correctly_ordered:
                direction = "DESC" if ranking_high else "ASC"
                errors.append(
                    "Per-group rankings must order the requested metric "
                    f"{direction} inside the partitioned ranking window."
                )
            elif not any(
                ranking["cutoff"] == requested for ranking in correctly_ordered
            ):
                errors.append(
                    f"The per-group ranking must filter its ROW_NUMBER/RANK result "
                    f"to {requested} row(s) per group."
                )
            return errors

        candidates = [
            *(
                {
                    "cutoff": ranking["cutoff"],
                    "descending": ranking["descending"],
                }
                for ranking in all_windows
                if not ranking["partitioned"] and ranking["has_order"]
            ),
            *_ordered_limit_rankings(expression),
        ]
        matching_cutoff = [
            candidate for candidate in candidates if candidate["cutoff"] == requested
        ]
        if not candidates:
            errors.append("A ranking question requires an explicit ORDER BY on its metric.")
        else:
            if not matching_cutoff:
                errors.append(
                    f"The ranking must select exactly {requested} row(s) with LIMIT "
                    f"{requested} or an equivalent window-rank cutoff."
                )
            direction_candidates = matching_cutoff or candidates
            if not any(
                (ranking_high and candidate["descending"] is True)
                or (ranking_low and candidate["descending"] is False)
                for candidate in direction_candidates
            ):
                direction = "DESC" if ranking_high else "ASC"
                errors.append(
                    f"The ranking must order its requested metric {direction} before "
                    f"selecting {requested} row(s)."
                )
    return errors


def _validate_set_operations(expression: exp.Expression, question: str) -> list[str]:
    errors: list[str] = []
    asks_for_deduplication = bool(
        re.search(r"\b(unique|distinct|deduplicat(?:e|ed|ion))\b", question, re.IGNORECASE)
    )
    for union in expression.find_all(exp.Union):
        left = list(union.left.selects)
        right = list(union.right.selects)
        if len(left) != len(right):
            errors.append(
                "UNION branches must return the same number of columns at the same grain."
            )
        if union.args.get("distinct") is not False and not asks_for_deduplication:
            errors.append(
                "UNION would silently remove identical rows. Use UNION ALL unless the "
                "question explicitly requests unique/deduplicated records."
            )
    return errors


def _humanized_identifier(identifier: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", identifier)
    value = re.sub(r"[_\W]+", " ", value).strip().casefold()
    return value


def _question_mentions_identifier(question: str, identifier: str) -> bool:
    normalized = " ".join(question.casefold().split())
    candidates = {identifier.casefold(), _humanized_identifier(identifier)}
    return any(
        candidate
        and re.search(
            rf"(?<![a-z0-9_]){re.escape(candidate)}(?![a-z0-9_])",
            normalized,
        )
        for candidate in candidates
    )


def _validate_explicit_identifiers(
    expression: exp.Expression,
    question: str,
    metadata: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    referenced_tables = {table.name.casefold() for table in expression.find_all(exp.Table)}
    referenced_columns = {column.name.casefold() for column in expression.find_all(exp.Column)}
    string_literals = {
        str(literal.this).casefold()
        for literal in expression.find_all(exp.Literal)
        if literal.is_string
    }
    for table, table_details in metadata.get("tables", {}).items():
        if _question_mentions_identifier(question, table) and table.casefold() not in referenced_tables:
            errors.append(
                f"The question explicitly names table '{table}', but SQL does not reference it."
            )
        for column in table_details.get("columns", {}):
            if (
                _question_mentions_identifier(question, column)
                and column.casefold() not in referenced_columns
            ):
                errors.append(
                    f"The question explicitly names column '{column}', but SQL does not use it."
                )
            for sample in table_details["columns"][column].get("sample_values") or []:
                if not isinstance(sample, str) or len(sample.strip()) < 2:
                    continue
                if (
                    _question_mentions_identifier(question, sample)
                    and sample.casefold() not in string_literals
                ):
                    errors.append(
                        f"The question explicitly names category value {sample!r}, but "
                        "SQL does not filter or otherwise use that exact value."
                    )
    for literal_date in set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", question)):
        if literal_date not in expression.sql(dialect="sqlite"):
            errors.append(
                f"The explicit date {literal_date} from the question is missing from SQL."
            )
    return errors


def validate_query_semantics(
    expression: exp.Expression,
    *,
    metadata: dict[str, Any],
    question: str = "",
) -> list[str]:
    """Return actionable semantic errors derived only from live metadata."""
    if not metadata.get("tables"):
        return ["No live table metadata is available for semantic validation."]
    try:
        qualify(
            expression.copy(),
            dialect="sqlite",
            schema=_schema(metadata),
            expand_stars=True,
            validate_qualify_columns=True,
            quote_identifiers=False,
            identify=False,
        )
    except SqlglotError as exc:
        return [f"Column resolution failed against the live schema: {exc}"]

    errors = [
        *_validate_joins(expression, metadata),
        *_validate_aggregate_types(expression, metadata),
        *_validate_fact_fanout(expression, metadata),
        *_validate_default_aggregations(expression, metadata, question),
        *_validate_grouping(expression),
        *_validate_set_operations(expression, question),
        *_validate_explicit_identifiers(expression, question, metadata),
        *_validate_question_alignment(expression, question, metadata),
    ]
    return list(dict.fromkeys(errors))
