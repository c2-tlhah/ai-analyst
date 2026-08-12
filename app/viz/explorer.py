"""Deterministic, on-demand charts for already retrieved query results.

Unlike the LLM visualization planner, this module is interactive: the UI asks
which chart types and time grains are valid for a concrete DataFrame, then
submits an explicit request. Every field is validated against the retrieved
columns before the controlled Plotly renderer is called.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import plotly.graph_objects as go

from app.llm.client import LLMClient, LLMError
from app.llm.schemas import InteractiveVisualizationPlan, VisualizationPlan
from app.viz.renderer import render_chart

ChartKind = Literal["bar", "line", "area", "scatter", "pie", "histogram", "box"]
Aggregation = Literal["none", "sum", "avg", "count", "min", "max"]
TimeGrain = Literal["none", "day", "week", "month", "quarter", "year"]

_CHART_LABELS = {
    "bar": "Bar",
    "line": "Line",
    "area": "Area",
    "scatter": "Scatter",
    "pie": "Pie",
    "histogram": "Histogram",
    "box": "Box",
}
_AGGREGATIONS = {"none", "sum", "avg", "count", "min", "max"}
_TIME_GRAINS = {"none", "day", "week", "month", "quarter", "year"}


@dataclass(frozen=True)
class ChartCapabilities:
    applicable: bool
    reason: str
    suggestions: tuple[str, ...] = ()
    chart_types: tuple[str, ...] = ()
    x_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    color_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    default_chart_type: str = "bar"
    default_x: str | None = None
    default_y: str | None = None
    default_aggregation: str = "none"


@dataclass(frozen=True)
class ChartBuildResult:
    figure: go.Figure | None = None
    error: str | None = None
    error_title: str | None = None
    suggestions: tuple[str, ...] = ()
    rows_plotted: int = 0
    ai_plan: InteractiveVisualizationPlan | None = None

    @property
    def ok(self) -> bool:
        return self.figure is not None and self.error is None


def chart_type_label(chart_type: str) -> str:
    return _CHART_LABELS.get(chart_type, chart_type.replace("_", " ").title())


def _chart_failure(
    reason: str,
    *suggestions: str,
    title: str = "This graph is not applicable",
) -> ChartBuildResult:
    return ChartBuildResult(
        error=reason,
        error_title=title,
        suggestions=tuple(suggestion for suggestion in suggestions if suggestion),
    )


def _available_chart_suggestion(capabilities: ChartCapabilities) -> str:
    labels = [chart_type_label(value) for value in capabilities.chart_types]
    return f"Try an applicable chart: {', '.join(labels)}." if labels else ""


def _not_applicable_reason(chart_type: str, capabilities: ChartCapabilities) -> tuple[str, str]:
    if chart_type == "scatter":
        return (
            "A scatter plot needs two different numeric columns, but this result has "
            f"{len(capabilities.numeric_columns)}.",
            "Ask a follow-up that retrieves two measures, such as SalesAmount and OrderQuantity.",
        )
    if chart_type == "histogram":
        return (
            "A histogram needs at least one numeric column, and this result has none.",
            "Ask for a numeric measure such as sales amount, quantity, price, or count.",
        )
    if chart_type == "pie":
        return (
            "A pie chart needs both a category and a numeric value in the retrieved result.",
            "Ask for a small category breakdown with a total, such as sales by channel.",
        )
    if chart_type in {"line", "area"}:
        return (
            f"A {chart_type_label(chart_type).lower()} chart needs an ordered X axis and a numeric value.",
            "For a time trend, ask for a date plus a measure, such as monthly sales.",
        )
    if chart_type == "box":
        return (
            "A box plot needs a numeric measure with multiple observations.",
            "Retrieve row-level numeric values, optionally with a category for comparison.",
        )
    return (
        f"{chart_type_label(chart_type)} cannot represent the columns in this result.",
        "Retrieve at least one category/date column and one numeric measure.",
    )


def _as_datetime(series: pd.Series) -> pd.Series | None:
    """Return parsed datetime values when a column is convincingly temporal."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    name = str(series.name).lower()
    if pd.api.types.is_numeric_dtype(series):
        if "year" not in name:
            return None
        parsed = pd.to_datetime(series.astype("Int64").astype(str), format="%Y", errors="coerce")
    else:
        non_empty = series.dropna().astype(str).str.strip()
        if non_empty.empty:
            return None
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(series, errors="coerce")

    populated = int(series.notna().sum())
    if populated == 0 or int(parsed.notna().sum()) / populated < 0.8:
        return None
    return parsed


def get_chart_capabilities(df: pd.DataFrame | None) -> ChartCapabilities:
    """Describe charts that are meaningful for a retrieved DataFrame."""
    if df is None:
        return ChartCapabilities(
            False,
            "There is no retrieved dataset available for graphing.",
            (
                "Run a data question successfully before opening the graph builder.",
                "Ask for a category or date together with a numeric measure.",
            ),
        )
    if df.empty:
        return ChartCapabilities(
            False,
            "The query returned zero rows, so there are no values to plot.",
            (
                "Remove or widen restrictive filters and run the query again.",
                "Check whether the requested date range exists in the data.",
            ),
        )
    if df.shape == (1, 1):
        return ChartCapabilities(
            False,
            "This result contains a single value; a graph needs multiple points.",
            (
                "Keep the value as a KPI/table instead of graphing it.",
                "Ask a follow-up such as 'break this down by month, product, or channel'.",
            ),
        )

    columns = [str(column) for column in df.columns]
    numeric = [str(column) for column in df.select_dtypes(include="number").columns]
    datetime_columns = [column for column in columns if _as_datetime(df[column]) is not None]
    color_columns = [
        column
        for column in columns
        if column not in numeric and column not in datetime_columns
    ]

    chart_types: list[str] = []
    if numeric:
        chart_types.append("histogram")
        if len(columns) > 1:
            chart_types.extend(["bar", "line", "area", "box"])
        if len(numeric) >= 2:
            chart_types.append("scatter")
        if color_columns:
            chart_types.append("pie")
    elif columns and len(df) > 1:
        # A categorical result can still be graphed as row counts per category.
        chart_types.append("bar")

    if not chart_types:
        return ChartCapabilities(
            False,
            "The retrieved columns do not contain a usable measure or repeated category.",
            (
                "Ask for a numeric measure such as sales, quantity, price, or count.",
                "Include a category or date breakdown to create multiple plotted points.",
            ),
            x_columns=tuple(columns),
            datetime_columns=tuple(datetime_columns),
        )

    if datetime_columns and numeric:
        default_type = "line"
        default_x = datetime_columns[0]
        default_y = numeric[0]
        default_aggregation = "sum"
    elif color_columns and numeric:
        default_type = "bar"
        default_x = color_columns[0]
        default_y = numeric[0]
        default_aggregation = "sum"
    elif len(numeric) >= 2:
        default_type = "scatter"
        default_x = numeric[0]
        default_y = numeric[1]
        default_aggregation = "none"
    elif numeric:
        default_type = "histogram"
        default_x = numeric[0]
        default_y = None
        default_aggregation = "none"
    else:
        default_type = "bar"
        default_x = columns[0]
        default_y = None
        default_aggregation = "count"

    ordered_types = [
        default_type,
        *[chart_type for chart_type in chart_types if chart_type != default_type],
    ]
    return ChartCapabilities(
        True,
        "This result can be visualized. Choose a chart and optionally group dates "
        "by week, month, quarter, or year.",
        chart_types=tuple(ordered_types),
        x_columns=tuple(columns),
        numeric_columns=tuple(numeric),
        color_columns=tuple(color_columns),
        datetime_columns=tuple(datetime_columns),
        default_chart_type=default_type,
        default_x=default_x,
        default_y=default_y,
        default_aggregation=default_aggregation,
    )


_AI_CHART_SYSTEM_PROMPT = """You plan a chart over data that has ALREADY been retrieved.
You never write SQL or Python and you never request new columns.

Hard requirements:
- Use exact column names from RESULT PROFILE only.
- chart_type is one of bar, line, area, scatter, pie, histogram, box.
- x, y, and color must be exact available column names or null where permitted.
- y must be numeric. For a row-count bar chart, use y=null and aggregation=count.
- Use line or area for a time trend, bar for category comparison/ranking, histogram
  for one numeric distribution, scatter for two numeric measures, and pie only for
  a small part-to-whole category result.
- time_grain is one of none, day, week, month, quarter, year. Use a non-none value
  only when x is listed as a datetime column.
- If the user asks for weekly/monthly/etc. grouping, reflect it in time_grain.
- Choose an aggregation whenever several rows may share an x/time bucket.
- Explain the mapping in one short sentence.
- Always include a non-empty title. aggregation and time_grain must never be null;
  use "none" when they do not apply. Use the exact response-schema key names.
"""


def _explicit_time_grain(request: str) -> str | None:
    """Extract an unambiguous user-requested time grouping as a safety net."""
    patterns = {
        "day": r"\b(daily|by day|per day)\b",
        "week": r"\b(weekly|by week|per week)\b",
        "month": r"\b(monthly|by month|per month)\b",
        "quarter": r"\b(quarterly|by quarter|per quarter)\b",
        "year": r"\b(yearly|annually|annual|by year|per year)\b",
    }
    lowered = request.lower()
    return next((grain for grain, pattern in patterns.items() if re.search(pattern, lowered)), None)


def generate_ai_exploratory_chart(
    llm_client: LLMClient,
    df: pd.DataFrame | None,
    *,
    request: str,
) -> ChartBuildResult:
    """Use an LLM to interpret a chart request, then validate/render deterministically."""
    capabilities = get_chart_capabilities(df)
    if not capabilities.applicable or df is None:
        return _chart_failure(capabilities.reason, *capabilities.suggestions)
    request = (request or "").strip()
    if not request:
        return _chart_failure(
            "No graph request was entered.",
            "Describe the chart type, axes, and grouping you want.",
            "Example: Show monthly TotalSales as a line chart.",
            title="Describe the graph you need",
        )

    profile = {
        "row_count": len(df),
        "columns": [{"name": str(column), "dtype": str(df[column].dtype)} for column in df.columns],
        "numeric_columns": list(capabilities.numeric_columns),
        "datetime_columns": list(capabilities.datetime_columns),
        "categorical_columns": list(capabilities.color_columns),
        "applicable_chart_types": list(capabilities.chart_types),
        "sample_rows": df.head(8).to_dict(orient="records"),
    }
    user_prompt = (
        f"USER CHART REQUEST:\n{request}\n\n"
        f"RESULT PROFILE (JSON):\n{json.dumps(profile, default=str)}"
    )
    try:
        plan = llm_client.complete_json(
            system_prompt=_AI_CHART_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=InteractiveVisualizationPlan,
        )
    except LLMError as exc:
        return _chart_failure(
            f"The selected AI model could not produce a valid graph plan: {exc}",
            "Retry once in case the model returned incomplete structured output.",
            "Use the manual graph controls below, which do not require an AI call.",
            "Select another model/provider if the failure repeats.",
            title="AI graph planning failed",
        )

    # Small local models sometimes describe the requested monthly/weekly grouping
    # correctly but leave time_grain="none" in their structured fields. Honor an
    # explicit phrase from the user's request and repair the rest of that mapping
    # from known-valid columns before the deterministic validator runs.
    requested_grain = _explicit_time_grain(request)
    if requested_grain is not None:
        updates: dict[str, object] = {"time_grain": requested_grain}
        if plan.x not in capabilities.datetime_columns and capabilities.datetime_columns:
            updates["x"] = capabilities.datetime_columns[0]
        if plan.y not in capabilities.numeric_columns and capabilities.numeric_columns:
            updates["y"] = capabilities.numeric_columns[0]
        if plan.chart_type not in {"line", "area", "bar"}:
            updates["chart_type"] = "line"
        if plan.aggregation == "none":
            selected_y = updates.get("y", plan.y)
            updates["aggregation"] = "sum" if selected_y is not None else "count"
        plan = plan.model_copy(update=updates)

    result = build_exploratory_chart(
        df,
        chart_type=plan.chart_type,
        x=plan.x,
        y=plan.y,
        color=plan.color,
        aggregation=plan.aggregation,
        time_grain=plan.time_grain,
        title=plan.title,
    )
    return ChartBuildResult(
        figure=result.figure,
        error=result.error,
        error_title=result.error_title,
        suggestions=result.suggestions,
        rows_plotted=result.rows_plotted,
        ai_plan=plan,
    )


def _time_bucket(values: pd.Series, grain: str) -> pd.Series:
    if grain == "day":
        return values.dt.floor("D")
    frequencies = {"week": "W-SUN", "month": "M", "quarter": "Q", "year": "Y"}
    return values.dt.to_period(frequencies[grain]).dt.start_time


def build_exploratory_chart(
    df: pd.DataFrame | None,
    *,
    chart_type: str,
    x: str | None,
    y: str | None,
    color: str | None = None,
    aggregation: str = "none",
    time_grain: str = "none",
    title: str = "Retrieved data",
) -> ChartBuildResult:
    """Validate and render a user-selected chart for retrieved data."""
    capabilities = get_chart_capabilities(df)
    if not capabilities.applicable or df is None:
        return _chart_failure(capabilities.reason, *capabilities.suggestions)
    if chart_type not in capabilities.chart_types:
        reason, required_data = _not_applicable_reason(chart_type, capabilities)
        return _chart_failure(
            reason,
            required_data,
            _available_chart_suggestion(capabilities),
        )
    if aggregation not in _AGGREGATIONS:
        return _chart_failure(
            f"'{aggregation}' is not a supported aggregation.",
            "Choose Sum, Average, Count, Minimum, Maximum, or No aggregation.",
            title="Unsupported aggregation",
        )
    if time_grain not in _TIME_GRAINS:
        return _chart_failure(
            f"'{time_grain}' is not a supported time grouping.",
            "Choose Day, Week, Month, Quarter, Year, or Original dates.",
            title="Unsupported time grouping",
        )

    columns = set(capabilities.x_columns)
    if x not in columns and not (chart_type == "box" and x is None):
        return _chart_failure(
            f"The selected X-axis column '{x}' is not in the retrieved result.",
            f"Choose one of: {', '.join(capabilities.x_columns)}.",
            title="Invalid X axis",
        )
    if y is not None and y not in capabilities.numeric_columns:
        available = ", ".join(capabilities.numeric_columns) or "none"
        return _chart_failure(
            f"The selected Y-axis column '{y}' is not numeric.",
            f"Available numeric columns: {available}.",
            "Use Count rows if you only have categorical columns.",
            title="Invalid Y axis",
        )
    if color is not None and color not in columns:
        return _chart_failure(
            f"The grouping column '{color}' is not in the retrieved result.",
            f"Choose one of: {', '.join(capabilities.x_columns)}.",
            "Select None if you do not need separate series.",
            title="Invalid group or color",
        )
    if color == x:
        color = None

    if chart_type == "histogram":
        if x not in capabilities.numeric_columns:
            return _chart_failure(
                f"A histogram cannot use '{x}' because it is not numeric.",
                f"Choose a numeric column: {', '.join(capabilities.numeric_columns)}.",
                "Use a bar chart to count categorical values instead.",
            )
        y = None
        aggregation = "none"
        time_grain = "none"
    elif chart_type == "scatter":
        if x not in capabilities.numeric_columns or y not in capabilities.numeric_columns:
            return _chart_failure(
                "A scatter plot requires two numeric columns.",
                f"Available numeric columns: {', '.join(capabilities.numeric_columns) or 'none'}.",
                "Ask a follow-up that retrieves another numeric measure.",
            )
        if x == y:
            return _chart_failure(
                f"Both scatter axes use '{x}', so the plot would not show a relationship.",
                "Choose two different numeric columns.",
                "Use a histogram instead if you want the distribution of one measure.",
            )
        aggregation = "none"
        time_grain = "none"
    elif chart_type == "box":
        if y not in capabilities.numeric_columns:
            return _chart_failure(
                "A box plot requires a numeric Y-axis column.",
                f"Choose one of: {', '.join(capabilities.numeric_columns) or 'no numeric columns available'}.",
                "Ask for row-level numeric values if the result only contains labels.",
            )
        aggregation = "none"
        time_grain = "none"
    elif y is None and aggregation != "count":
        return _chart_failure(
            "This chart has no numeric Y value and is not configured to count rows.",
            "Choose a numeric Y-axis column.",
            "Or leave Y as Count rows and select Count as the aggregation.",
            title="A value or row count is required",
        )

    data = df.copy()
    if time_grain != "none":
        if x not in capabilities.datetime_columns:
            dates = ", ".join(capabilities.datetime_columns) or "none"
            return _chart_failure(
                f"'{x}' is not a date column, so it cannot be grouped by {time_grain}.",
                f"Available date columns: {dates}.",
                "Choose Original dates, or ask a follow-up that includes OrderDate.",
                title="Time grouping is not applicable",
            )
        parsed = _as_datetime(data[x])
        if parsed is None:
            return _chart_failure(
                f"Values in '{x}' could not be reliably parsed as dates.",
                "Choose another date column or use Original dates.",
                "Ask a follow-up that returns a valid date field such as OrderDate.",
                title="Invalid date values",
            )
        data[x] = _time_bucket(parsed, time_grain)
        data = data[data[x].notna()]
        if data.empty:
            return _chart_failure(
                "No valid dates remained after applying the requested time grouping.",
                "Widen the query's date range or remove null/invalid dates.",
                "Use Original dates to inspect the raw values first.",
                title="No dates available to plot",
            )

    group_columns = [x] if x is not None else []
    if color and color != x:
        group_columns.append(color)

    should_aggregate = aggregation != "none" and chart_type not in {
        "histogram", "scatter", "box"
    }
    if should_aggregate:
        if aggregation == "count":
            if y is None:
                data = data.groupby(group_columns, dropna=False).size().reset_index(name="Row count")
                y = "Row count"
            else:
                data = data.groupby(group_columns, dropna=False)[y].count().reset_index()
        else:
            if y is None:
                return _chart_failure(
                    f"{aggregation.title()} requires a numeric Y-axis column.",
                    f"Choose one of: {', '.join(capabilities.numeric_columns) or 'none available'}.",
                    "Use Count when graphing categories without a numeric measure.",
                    title="Aggregation is not applicable",
                )
            pandas_aggregation = "mean" if aggregation == "avg" else aggregation
            data = data.groupby(group_columns, dropna=False)[y].agg(pandas_aggregation).reset_index()
    elif time_grain != "none" and data.duplicated(subset=group_columns).any():
        return _chart_failure(
            f"Multiple rows fall into each {time_grain} bucket and no aggregation was selected.",
            "Select Sum for totals, Average for typical values, or Count for frequency.",
            "Use Original dates if each row should remain a separate point.",
            title="Time buckets need aggregation",
        )

    plan = VisualizationPlan(
        chart_type=chart_type,
        title=(title or "Retrieved data").strip(),
        x=x,
        y=y,
        color=color,
        agg="none",
    )
    figure = render_chart(plan, data)
    if figure is None:
        return _chart_failure(
            "Plotly could not render the selected chart with these columns and options.",
            _available_chart_suggestion(capabilities),
            "Remove the group/color field or choose the recommended axes.",
            title="Graph rendering failed",
        )
    return ChartBuildResult(figure=figure, rows_plotted=len(data))
