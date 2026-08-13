import pandas as pd

from app.analysis.insights import (
    MAX_PREVIEW_ROWS,
    deterministic_direct_answer,
    enforce_numeric_grounding,
    summarize_dataframe,
)
from app.llm.schemas import InsightResult


def test_result_summary_sends_only_five_raw_rows_to_the_llm():
    dataframe = pd.DataFrame(
        {
            "ProductName": [f"Product {index}" for index in range(100)],
            "SalesAmount": list(range(100)),
        }
    )

    summary = summarize_dataframe(dataframe)

    assert MAX_PREVIEW_ROWS == 5
    assert summary["row_count"] == 100
    assert len(summary["preview"]) == 5
    assert summary["numeric_summary"]["SalesAmount"]["max"] == 99


def test_scalar_answer_is_built_from_the_executed_cell():
    answer = deterministic_direct_answer(
        "What is the total?", pd.DataFrame({"total_value": [123.5]})
    )
    assert answer is not None
    assert answer.summary == "Result: total_value = 123.5."


def test_ranking_answer_uses_first_executed_row_not_model_prose():
    dataframe = pd.DataFrame(
        {"item": ["Actual winner", "Runner up"], "score": [42, 30]}
    )
    answer = deterministic_direct_answer("Which item has the highest score?", dataframe)
    assert answer is not None
    assert "Actual winner" in answer.summary
    assert "42" in answer.summary
    assert "Runner up" not in answer.summary


def test_partitioned_ranking_describes_rows_as_per_group_winners():
    dataframe = pd.DataFrame(
        {
            "month": ["2026-01", "2026-02"],
            "item": ["A", "B"],
            "units": [42, 30],
        }
    )
    answer = deterministic_direct_answer(
        "Which item sold most by month?",
        dataframe,
        sql=(
            "SELECT month, item, units, ROW_NUMBER() OVER "
            "(PARTITION BY month ORDER BY units DESC) AS rn FROM sales"
        ),
    )

    assert answer is not None
    assert "winner for 2 group(s)" in answer.summary
    assert not any(finding.startswith("#") for finding in answer.key_findings)


def test_non_direct_multirow_analysis_keeps_model_summary_path():
    dataframe = pd.DataFrame({"month": ["Jan", "Feb"], "value": [1, 2]})
    assert deterministic_direct_answer("Show the monthly trend", dataframe) is None


def test_unsupported_model_number_is_replaced_by_executed_evidence():
    dataframe = pd.DataFrame({"month": ["Jan", "Feb"], "value": [10, 20]})
    grounded = enforce_numeric_grounding(
        InsightResult(summary="The value reached 999.", key_findings=[]),
        question="Show the monthly trend",
        df=dataframe,
    )
    assert "999" not in grounded.summary
    assert "10" in grounded.summary


def test_supported_model_numbers_are_preserved():
    dataframe = pd.DataFrame({"month": ["Jan", "Feb"], "value": [10, 20]})
    insight = InsightResult(summary="The value rose from 10 to 20.", key_findings=[])
    assert enforce_numeric_grounding(
        insight, question="Show the monthly trend", df=dataframe
    ) == insight
