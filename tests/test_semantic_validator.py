from app.sql.validator import validate_sql


def _column(sql_type, role, *, family=None, observed=None):
    return {
        "sql_type": sql_type,
        "semantic_role": role,
        "declared_type_family": family,
        "observed_value_family": observed,
    }


def _metadata():
    return {
        "tables": {
            "accounts": {
                "kind": "dimension",
                "columns": {
                    "account_id": _column("INTEGER", "key", family="numeric"),
                    "segment": _column("TEXT", "categorical_attribute", family="text"),
                },
            },
            "activity_log": {
                "kind": "fact",
                "columns": {
                    "activity_id": _column("INTEGER", "key", family="numeric"),
                    "account_id": _column("INTEGER", "key", family="numeric"),
                    "occurred_at": _column("TEXT", "temporal", family="text"),
                    "metric_value": _column("REAL", "measure", family="numeric"),
                    "state_label": _column("TEXT", "categorical_attribute", family="text"),
                },
            },
            "unrelated_notes": {
                "kind": "unknown",
                "columns": {
                    "note_id": _column("INTEGER", "key", family="numeric"),
                    "body": _column("TEXT", "categorical_attribute", family="text"),
                },
            },
        },
        "relationships": [
            {
                "from_table": "activity_log",
                "from_column": "account_id",
                "to_table": "accounts",
                "to_column": "account_id",
                "source": "declared",
            }
        ],
    }


def _validate(sql, question=""):
    metadata = _metadata()
    return validate_sql(
        sql,
        set(metadata["tables"]),
        max_rows=100,
        metadata=metadata,
        question=question,
    )


def test_live_schema_column_resolution_rejects_hallucinated_column():
    result = _validate("SELECT imaginary_metric FROM activity_log")
    assert not result.is_valid
    assert any("could not be resolved" in error for error in result.errors)


def test_verified_relationship_join_and_numeric_aggregation_are_accepted():
    result = _validate(
        "SELECT a.segment, SUM(e.metric_value) AS total_value "
        "FROM activity_log e JOIN accounts a ON e.account_id = a.account_id "
        "GROUP BY a.segment ORDER BY total_value DESC LIMIT 5",
        "top 5 segments by total value",
    )
    assert result.is_valid, result.errors


def test_join_on_undocumented_columns_is_rejected():
    result = _validate(
        "SELECT * FROM activity_log e JOIN accounts a "
        "ON e.activity_id = a.account_id"
    )
    assert not result.is_valid
    assert any("verified inferred relationship" in error for error in result.errors)


def test_join_between_unrelated_tables_is_rejected():
    result = _validate(
        "SELECT * FROM activity_log e JOIN unrelated_notes n "
        "ON e.activity_id = n.note_id"
    )
    assert not result.is_valid
    assert any("relationship" in error for error in result.errors)


def test_sum_of_text_label_is_rejected_before_execution():
    result = _validate("SELECT SUM(state_label) FROM activity_log")
    assert not result.is_valid
    assert any("non-numeric" in error for error in result.errors)


def test_sum_of_numeric_identifier_is_rejected_before_execution():
    result = _validate("SELECT SUM(activity_id) FROM activity_log")

    assert not result.is_valid
    assert any("non-numeric" in error for error in result.errors)


def test_declared_numeric_column_with_observed_mixed_values_is_rejected():
    metadata = _metadata()
    metadata["tables"]["activity_log"]["columns"]["metric_value"][
        "observed_value_family"
    ] = "mixed"
    result = validate_sql(
        "SELECT SUM(metric_value) FROM activity_log",
        set(metadata["tables"]),
        100,
        metadata=metadata,
    )
    assert not result.is_valid
    assert any("non-numeric" in error for error in result.errors)


def test_sqlites_arbitrary_non_grouped_projection_is_rejected():
    result = _validate(
        "SELECT state_label, SUM(metric_value) FROM activity_log"
    )
    assert not result.is_valid
    assert any("arbitrary value" in error for error in result.errors)


def test_question_ranking_count_and_direction_are_repaired_when_unambiguous():
    result = _validate(
        "SELECT state_label, SUM(metric_value) AS value FROM activity_log "
        "GROUP BY state_label ORDER BY value ASC LIMIT 10",
        "top 5 states by value",
    )
    assert result.is_valid, result.errors
    assert "ORDER BY value DESC" in result.sanitized_sql
    assert "LIMIT 5" in result.sanitized_sql
    assert len(result.repairs) == 2


def test_scalar_maximum_is_an_extrema_query_not_a_ranked_row_query():
    result = _validate(
        "SELECT MAX(metric_value) AS highest_value FROM activity_log",
        "What is the highest metric value?",
    )

    assert result.is_valid, result.errors


def test_grouped_maximum_does_not_require_a_global_limit():
    result = _validate(
        "SELECT state_label, MAX(metric_value) AS highest_value "
        "FROM activity_log GROUP BY state_label ORDER BY state_label",
        "What is the highest metric value by state label?",
    )

    assert result.is_valid, result.errors


def test_global_ranking_inside_cte_can_be_reordered_for_display():
    result = _validate(
        "WITH winners AS ("
        "SELECT state_label, SUM(metric_value) AS value FROM activity_log "
        "GROUP BY state_label ORDER BY value DESC LIMIT 2"
        ") SELECT state_label, value FROM winners ORDER BY state_label",
        "Show the top 2 states by metric value",
    )

    assert result.is_valid, result.errors


def test_global_window_ranking_is_accepted_without_top_level_limit():
    result = _validate(
        "WITH totals AS ("
        "SELECT state_label, SUM(metric_value) AS value FROM activity_log "
        "GROUP BY state_label"
        "), ranked AS ("
        "SELECT state_label, value, ROW_NUMBER() OVER (ORDER BY value DESC) AS rn "
        "FROM totals"
        ") SELECT state_label, value FROM ranked WHERE rn <= 2",
        "Show the top 2 states by metric value",
    )

    assert result.is_valid, result.errors


def test_threshold_wording_is_not_misclassified_as_a_ranking():
    result = _validate(
        "SELECT state_label, SUM(metric_value) AS value FROM activity_log "
        "GROUP BY state_label HAVING SUM(metric_value) >= 10",
        "Show states with at least 10 total metric value",
    )

    assert result.is_valid, result.errors


def test_singular_ranking_is_locally_normalized_to_one_row():
    result = _validate(
        "SELECT state_label, SUM(metric_value) AS value FROM activity_log "
        "GROUP BY state_label ORDER BY value DESC LIMIT 10",
        "which state has the highest value",
    )
    assert result.is_valid, result.errors
    assert "ORDER BY value DESC" in result.sanitized_sql
    assert "LIMIT 1" in result.sanitized_sql
    assert len(result.repairs) == 1


def test_ambiguous_multiple_ranking_metrics_are_not_guessed_or_repaired():
    result = _validate(
        "SELECT state_label, SUM(metric_value) AS value, COUNT(*) AS records "
        "FROM activity_log GROUP BY state_label",
        "Show the top 5 states",
    )

    assert not result.is_valid
    assert result.repairs == []
    assert any("ORDER BY" in error for error in result.errors)


def test_partitioned_monthly_winner_does_not_require_global_limit_or_metric_sort():
    result = _validate(
        """WITH totals AS (
          SELECT strftime('%Y-%m', occurred_at) AS sales_month,
                 state_label,
                 SUM(metric_value) AS total_value
          FROM activity_log
          GROUP BY strftime('%Y-%m', occurred_at), state_label
        ), ranked AS (
          SELECT sales_month, state_label, total_value,
                 ROW_NUMBER() OVER (
                   PARTITION BY sales_month ORDER BY total_value DESC
                 ) AS rn
          FROM totals
        )
        SELECT sales_month, state_label, total_value
        FROM ranked WHERE rn = 1 ORDER BY sales_month""",
        "what is the highest state by month",
    )

    assert result.is_valid, result.errors


def test_partitioned_ranking_recognizes_in_each_group_wording():
    result = _validate(
        """WITH totals AS (
          SELECT strftime('%Y-%m', occurred_at) AS sales_month,
                 state_label,
                 SUM(metric_value) AS total_value
          FROM activity_log
          GROUP BY strftime('%Y-%m', occurred_at), state_label
        ), ranked AS (
          SELECT sales_month, state_label, total_value,
                 ROW_NUMBER() OVER (
                   PARTITION BY sales_month ORDER BY total_value DESC
                 ) AS rn
          FROM totals
        )
        SELECT sales_month, state_label, total_value
        FROM ranked WHERE rn = 1 ORDER BY sales_month""",
        "Top state in each month in 2013",
    )

    assert result.is_valid, result.errors


def test_unambiguous_global_ranking_is_repaired_without_an_llm_retry():
    result = _validate(
        "SELECT state_label, SUM(metric_value) AS total_value "
        "FROM activity_log GROUP BY state_label",
        "Show the top 3 states by metric value",
    )

    assert result.is_valid, result.errors
    assert "ORDER BY total_value DESC" in result.sanitized_sql
    assert "LIMIT 3" in result.sanitized_sql
    assert len(result.repairs) == 2


def test_partitioned_monthly_winner_requires_correct_window_direction():
    result = _validate(
        """WITH totals AS (
          SELECT strftime('%Y-%m', occurred_at) AS sales_month,
                 state_label,
                 SUM(metric_value) AS total_value
          FROM activity_log
          GROUP BY strftime('%Y-%m', occurred_at), state_label
        ), ranked AS (
          SELECT sales_month, state_label, total_value,
                 ROW_NUMBER() OVER (
                   PARTITION BY sales_month ORDER BY total_value ASC
                 ) AS rn
          FROM totals
        )
        SELECT sales_month, state_label, total_value
        FROM ranked WHERE rn = 1 ORDER BY sales_month""",
        "what is the highest state by month",
    )

    assert not result.is_valid
    assert any("partitioned ranking window" in error for error in result.errors)


def test_union_does_not_silently_deduplicate_event_rows():
    result = _validate(
        "SELECT state_label FROM activity_log "
        "UNION SELECT body FROM unrelated_notes"
    )
    assert not result.is_valid
    assert any("UNION ALL" in error for error in result.errors)


def test_explicit_average_question_requires_avg():
    result = _validate(
        "SELECT SUM(metric_value) FROM activity_log",
        "What is the average metric value?",
    )
    assert not result.is_valid
    assert any("AVG" in error for error in result.errors)


def test_how_many_requires_count():
    result = _validate("SELECT activity_id FROM activity_log", "How many activities?")

    assert not result.is_valid
    assert any("COUNT" in error for error in result.errors)


def test_total_of_named_numeric_measure_cannot_be_replaced_by_count():
    result = _validate(
        "SELECT COUNT(metric_value) FROM activity_log",
        "What is the total metric value?",
    )

    assert not result.is_valid
    assert any("SUM" in error for error in result.errors)


def test_explicit_sum_cannot_be_replaced_by_count():
    result = _validate(
        "SELECT COUNT(metric_value) FROM activity_log",
        "What is the sum of metric value?",
    )

    assert not result.is_valid
    assert any("SUM" in error for error in result.errors)


def test_unique_rows_require_deduplication():
    rejected = _validate("SELECT state_label FROM activity_log", "Show unique states")
    accepted = _validate(
        "SELECT DISTINCT state_label FROM activity_log", "Show unique states"
    )

    assert not rejected.is_valid
    assert accepted.is_valid, accepted.errors


def test_reused_aliases_in_union_branches_keep_local_column_types():
    result = _validate(
        "SELECT SUM(a.metric_value) AS value FROM activity_log a "
        "UNION ALL SELECT SUM(a.body) AS value FROM unrelated_notes a"
    )

    assert not result.is_valid
    assert any("non-numeric" in error for error in result.errors)


def test_aggregate_rejects_multiple_raw_fact_tables_in_one_scope():
    metadata = _metadata()
    metadata["tables"]["billing_log"] = {
        "kind": "fact",
        "columns": {
            "account_id": _column("INTEGER", "key", family="numeric"),
            "billed_value": _column("REAL", "measure", family="numeric"),
        },
    }
    metadata["relationships"].append(
        {
            "from_table": "billing_log",
            "from_column": "account_id",
            "to_table": "accounts",
            "to_column": "account_id",
            "source": "declared",
        }
    )
    result = validate_sql(
        "SELECT a.segment, SUM(e.metric_value), SUM(b.billed_value) "
        "FROM activity_log e "
        "JOIN accounts a ON a.account_id = e.account_id "
        "JOIN billing_log b ON b.account_id = a.account_id "
        "GROUP BY a.segment",
        set(metadata["tables"]),
        100,
        metadata=metadata,
    )

    assert not result.is_valid
    assert any("multiply rows" in error for error in result.errors)


def test_separately_aggregated_fact_sources_can_be_combined_safely():
    metadata = _metadata()
    metadata["tables"]["billing_log"] = {
        "kind": "fact",
        "columns": {
            "account_id": _column("INTEGER", "key", family="numeric"),
            "billed_value": _column("REAL", "measure", family="numeric"),
        },
    }
    result = validate_sql(
        "WITH activity AS ("
        "SELECT account_id, SUM(metric_value) AS value FROM activity_log GROUP BY account_id"
        "), billing AS ("
        "SELECT account_id, SUM(billed_value) AS value FROM billing_log GROUP BY account_id"
        ") SELECT account_id, value FROM activity "
        "UNION ALL SELECT account_id, value FROM billing",
        set(metadata["tables"]),
        100,
        metadata=metadata,
    )

    assert result.is_valid, result.errors


def test_profiled_default_aggregation_is_enforced_when_wording_is_ambiguous():
    metadata = _metadata()
    metadata["tables"]["activity_log"]["columns"]["metric_value"][
        "default_aggregation"
    ] = "avg"
    result = validate_sql(
        "SELECT state_label, SUM(metric_value) FROM activity_log GROUP BY state_label",
        set(metadata["tables"]),
        100,
        metadata=metadata,
        question="Show metric value by state label",
    )

    assert not result.is_valid
    assert any("defines AVG" in error for error in result.errors)


def test_count_cannot_silently_replace_a_profiled_measure_aggregation():
    metadata = _metadata()
    metadata["tables"]["activity_log"]["columns"]["metric_value"][
        "default_aggregation"
    ] = "sum"
    result = validate_sql(
        "SELECT state_label, COUNT(metric_value) FROM activity_log GROUP BY state_label",
        set(metadata["tables"]),
        100,
        metadata=metadata,
        question="Show metric value by state label",
    )

    assert not result.is_valid
    assert any("explicitly ask for a count" in error for error in result.errors)


def test_explicit_count_can_count_non_null_measure_values():
    metadata = _metadata()
    metadata["tables"]["activity_log"]["columns"]["metric_value"][
        "default_aggregation"
    ] = "sum"
    result = validate_sql(
        "SELECT state_label, COUNT(metric_value) FROM activity_log GROUP BY state_label",
        set(metadata["tables"]),
        100,
        metadata=metadata,
        question="Count non-null metric value records by state label",
    )

    assert result.is_valid, result.errors


def test_explicit_aggregation_can_override_profiled_default():
    metadata = _metadata()
    metadata["tables"]["activity_log"]["columns"]["metric_value"][
        "default_aggregation"
    ] = "avg"
    result = validate_sql(
        "SELECT state_label, SUM(metric_value) FROM activity_log GROUP BY state_label",
        set(metadata["tables"]),
        100,
        metadata=metadata,
        question="Show total metric value by state label",
    )

    assert result.is_valid, result.errors


def test_cte_output_can_join_a_real_dimension_after_inner_join_validation():
    result = _validate(
        "WITH totals AS ("
        "SELECT account_id, SUM(metric_value) AS value FROM activity_log "
        "GROUP BY account_id"
        ") SELECT a.segment, t.value FROM totals t "
        "JOIN accounts a ON a.account_id = t.account_id"
    )

    assert result.is_valid, result.errors


def test_composite_relationship_requires_every_key_part():
    metadata = {
        "tables": {
            "parents": {
                "kind": "dimension",
                "columns": {
                    "tenant_id": _column("INTEGER", "key", family="numeric"),
                    "entity_id": _column("INTEGER", "key", family="numeric"),
                },
            },
            "children": {
                "kind": "fact",
                "columns": {
                    "tenant_id": _column("INTEGER", "key", family="numeric"),
                    "entity_id": _column("INTEGER", "key", family="numeric"),
                    "value": _column("REAL", "measure", family="numeric"),
                },
            },
        },
        "relationships": [
            {
                "from_table": "children", "from_column": "tenant_id",
                "to_table": "parents", "to_column": "tenant_id",
                "constraint_id": 0, "constraint_sequence": 0, "constraint_size": 2,
            },
            {
                "from_table": "children", "from_column": "entity_id",
                "to_table": "parents", "to_column": "entity_id",
                "constraint_id": 0, "constraint_sequence": 1, "constraint_size": 2,
            },
        ],
    }
    partial = validate_sql(
        "SELECT * FROM children c JOIN parents p ON c.tenant_id = p.tenant_id",
        set(metadata["tables"]), 100, metadata=metadata,
    )
    complete = validate_sql(
        "SELECT * FROM children c JOIN parents p "
        "ON c.tenant_id = p.tenant_id AND c.entity_id = p.entity_id",
        set(metadata["tables"]), 100, metadata=metadata,
    )
    assert not partial.is_valid
    assert complete.is_valid, complete.errors


def test_explicit_live_column_cannot_be_substituted_with_another_measure():
    result = _validate(
        "SELECT SUM(activity_id) FROM activity_log",
        "What is the total metric value?",
    )
    assert not result.is_valid
    assert any("metric_value" in error for error in result.errors)


def test_explicit_date_literal_must_be_preserved():
    result = _validate(
        "SELECT COUNT(*) FROM activity_log WHERE occurred_at >= '2026-02-01'",
        "Count activity since 2026-01-01",
    )
    assert not result.is_valid
    assert any("2026-01-01" in error for error in result.errors)


def test_explicit_profiled_category_value_must_be_preserved():
    metadata = _metadata()
    metadata["tables"]["activity_log"]["columns"]["state_label"][
        "sample_values"
    ] = ["complete", "pending"]
    result = validate_sql(
        "SELECT COUNT(*) FROM activity_log WHERE state_label = 'pending'",
        set(metadata["tables"]),
        100,
        metadata=metadata,
        question="Count complete activity",
    )
    assert not result.is_valid
    assert any("'complete'" in error for error in result.errors)
