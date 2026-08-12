import pandas as pd

from app.llm.client import LLMClient
from app.llm.schemas import InteractiveVisualizationPlan
from app.viz.explorer import (
    build_exploratory_chart,
    generate_ai_exploratory_chart,
    get_chart_capabilities,
)


class ChartPlanningLLM(LLMClient):
    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return ""

    def complete_json(self, *, system_prompt, user_prompt, schema, max_repair_attempts=1):
        assert schema is InteractiveVisualizationPlan
        assert "Show monthly sales" in user_prompt
        return InteractiveVisualizationPlan(
            chart_type="line",
            title="Monthly sales",
            x="order_date",
            y="sales",
            color=None,
            aggregation="sum",
            time_grain="month",
            explanation="Sum sales into monthly points and connect them as a trend.",
        )


class ImperfectLocalChartLLM(ChartPlanningLLM):
    def complete_json(self, *, system_prompt, user_prompt, schema, max_repair_attempts=1):
        return InteractiveVisualizationPlan(
            chart_type="histogram",
            title="Monthly sales",
            x="sales",
            y=None,
            aggregation="none",
            time_grain="none",
            explanation="Show monthly sales.",
        )


def test_temporal_result_recommends_line_chart():
    dataframe = pd.DataFrame(
        {
            "order_date": ["2025-01-01", "2025-01-02", "2025-02-01"],
            "sales": [10.0, 20.0, 30.0],
        }
    )

    capabilities = get_chart_capabilities(dataframe)

    assert capabilities.applicable is True
    assert capabilities.default_chart_type == "line"
    assert capabilities.default_x == "order_date"
    assert capabilities.default_y == "sales"
    assert "order_date" in capabilities.datetime_columns


def test_monthly_chart_aggregates_retrieved_rows():
    dataframe = pd.DataFrame(
        {
            "order_date": ["2025-01-01", "2025-01-15", "2025-02-01"],
            "sales": [10.0, 20.0, 30.0],
        }
    )

    result = build_exploratory_chart(
        dataframe,
        chart_type="line",
        x="order_date",
        y="sales",
        aggregation="sum",
        time_grain="month",
        title="Monthly sales",
    )

    assert result.ok is True
    assert result.rows_plotted == 2
    assert list(result.figure.data[0].y) == [30.0, 30.0]


def test_scalar_result_explains_that_graph_is_not_applicable():
    capabilities = get_chart_capabilities(pd.DataFrame({"total_sales": [100.0]}))

    assert capabilities.applicable is False
    assert "single value" in capabilities.reason
    assert any("break this down" in suggestion for suggestion in capabilities.suggestions)


def test_time_grouping_rejects_non_date_column():
    dataframe = pd.DataFrame({"product": ["A", "B"], "sales": [10.0, 20.0]})

    result = build_exploratory_chart(
        dataframe,
        chart_type="bar",
        x="product",
        y="sales",
        aggregation="sum",
        time_grain="month",
    )

    assert result.ok is False
    assert "not a date column" in result.error
    assert result.error_title == "Time grouping is not applicable"
    assert any("OrderDate" in suggestion for suggestion in result.suggestions)


def test_inapplicable_scatter_explains_required_data_and_alternatives():
    dataframe = pd.DataFrame({"product": ["A", "B"], "sales": [10.0, 20.0]})

    result = build_exploratory_chart(
        dataframe,
        chart_type="scatter",
        x="sales",
        y="sales",
    )

    assert result.ok is False
    assert "needs two different numeric columns" in result.error
    assert any("two measures" in suggestion for suggestion in result.suggestions)
    assert any("applicable chart" in suggestion for suggestion in result.suggestions)


def test_empty_result_suggests_fixing_filters():
    capabilities = get_chart_capabilities(pd.DataFrame(columns=["month", "sales"]))

    assert capabilities.applicable is False
    assert "zero rows" in capabilities.reason
    assert any("filters" in suggestion for suggestion in capabilities.suggestions)


def test_categorical_data_can_be_graphed_as_row_counts():
    dataframe = pd.DataFrame({"channel": ["Internet", "Internet", "Reseller"]})

    capabilities = get_chart_capabilities(dataframe)
    result = build_exploratory_chart(
        dataframe,
        chart_type="bar",
        x="channel",
        y=None,
        aggregation="count",
    )

    assert capabilities.applicable is True
    assert result.ok is True
    assert result.rows_plotted == 2


def test_ai_chart_request_is_planned_then_validated_and_rendered():
    dataframe = pd.DataFrame(
        {
            "order_date": ["2025-01-01", "2025-01-15", "2025-02-01"],
            "sales": [10.0, 20.0, 30.0],
        }
    )

    result = generate_ai_exploratory_chart(
        ChartPlanningLLM(),
        dataframe,
        request="Show monthly sales as a line chart",
    )

    assert result.ok is True
    assert result.rows_plotted == 2
    assert result.ai_plan is not None
    assert result.ai_plan.time_grain == "month"
    assert "monthly points" in result.ai_plan.explanation


def test_explicit_monthly_request_repairs_an_imperfect_small_model_plan():
    dataframe = pd.DataFrame(
        {
            "order_date": ["2025-01-01", "2025-01-15", "2025-02-01"],
            "sales": [10.0, 20.0, 30.0],
        }
    )

    result = generate_ai_exploratory_chart(
        ImperfectLocalChartLLM(),
        dataframe,
        request="Show monthly sales as a graph",
    )

    assert result.ok is True
    assert result.rows_plotted == 2
    assert result.ai_plan.time_grain == "month"
    assert result.ai_plan.x == "order_date"
    assert result.ai_plan.y == "sales"
    assert result.ai_plan.chart_type == "line"
