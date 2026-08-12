from app.llm.schemas import (
    InsightResult,
    IntentResult,
    InteractiveVisualizationPlan,
    VisualizationPlan,
)


def test_intent_accepts_classification_alias_and_supplies_summary():
    result = IntentResult.model_validate(
        {
            "classification": "aggregate",
            "relevant_tables": ["FactInternetSales", "FactResellerSales"],
        }
    )

    assert result.analysis_type == "aggregation"
    assert result.intent_summary


def test_insight_accepts_common_model_aliases():
    result = InsightResult.model_validate(
        {"insight": "Sales increased.", "findings": "The west region led."}
    )

    assert result.summary == "Sales increased."
    assert result.key_findings == ["The west region led."]


def test_visualization_plan_supplies_missing_title():
    result = VisualizationPlan.model_validate(
        {"chart_type": "bar", "x": "ProductName", "y": "TotalSales", "agg": None}
    )

    assert result.title == "Analysis result"
    assert result.agg == "none"


def test_interactive_plan_repairs_null_aggregation_and_missing_title():
    result = InteractiveVisualizationPlan.model_validate(
        {
            "chart_type": "histogram",
            "x": "TotalSales",
            "y": None,
            "aggregation": None,
            "time_grain": None,
            "rationale": "Show the distribution.",
        }
    )

    assert result.title == "Retrieved data"
    assert result.aggregation == "none"
    assert result.time_grain == "none"
    assert result.explanation == "Show the distribution."
