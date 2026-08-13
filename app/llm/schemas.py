"""Pydantic schemas for every structured LLM output in the pipeline.

The LLM never free-forms SQL execution, chart code, or anything else with
side effects -- it only ever returns one of these typed objects, which the
deterministic backend then validates and acts on.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

AnalysisType = Literal[
    "aggregation", "trend", "comparison", "ranking", "lookup", "distribution", "other"
]


def _normalized_analysis_type(value: Any) -> str:
    """Map common model synonyms to the workflow's closed intent vocabulary."""
    normalized = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "aggregate": "aggregation",
        "aggregation": "aggregation",
        "trend_analysis": "trend",
        "time_series": "trend",
        "compare": "comparison",
        "rank": "ranking",
        "top_n": "ranking",
        "list": "lookup",
        "search": "lookup",
        "histogram": "distribution",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "aggregation", "trend", "comparison", "ranking", "lookup",
        "distribution", "other",
    }
    return normalized if normalized in allowed else "other"


class IntentResult(BaseModel):
    """Output of the intent-understanding node."""

    intent_summary: str = Field(description="One-sentence restatement of what the user is asking.")
    analysis_type: AnalysisType = Field(description="The shape of analysis being requested.")
    relevant_tables: list[str] = Field(
        default_factory=list,
        description="Best-guess table names (from the catalog given) that are relevant.",
    )
    metrics: list[str] = Field(
        default_factory=list, description="Business metrics/measures mentioned or implied."
    )
    filters_mentioned: list[str] = Field(
        default_factory=list, description="Filter conditions mentioned in the question."
    )
    time_range_mentioned: Optional[str] = Field(
        default=None, description="Any date/time range mentioned, in free text."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_model_synonyms(cls, value: Any) -> Any:
        """Accept predictable aliases emitted by smaller hosted/local models."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        classification = data.get("analysis_type", data.get("classification"))
        data["analysis_type"] = _normalized_analysis_type(classification)
        if not data.get("intent_summary"):
            data["intent_summary"] = (
                data.get("summary")
                or data.get("intent")
                or f"{data['analysis_type'].replace('_', ' ').title()} analysis request."
            )
        return data


class SQLGenerationResult(BaseModel):
    """Output of the SQL-generation node."""

    sql: str = Field(description="A single read-only SQLite SELECT statement.")
    explanation: str = Field(description="One or two sentences on what the query computes.")
    tables_used: list[str] = Field(default_factory=list)


ChartType = Literal["bar", "line", "scatter", "pie", "histogram", "box", "area", "table", "none"]
AggFunction = Literal["sum", "avg", "count", "min", "max", "none"]
InteractiveChartType = Literal["bar", "line", "scatter", "pie", "histogram", "box", "area"]
TimeGrain = Literal["none", "day", "week", "month", "quarter", "year"]


class VisualizationPlan(BaseModel):
    """Output of the visualization-planning node.

    This is a *plan*, not code: the backend's controlled renderer
    (:mod:`app.viz.renderer`) is the only thing that turns it into an
    actual chart.
    """

    chart_type: ChartType = Field(description="Kind of chart to draw, or 'table'/'none'.")
    title: str = Field(description="Short chart title.")
    x: Optional[str] = Field(default=None, description="Column name to use for the X axis.")
    y: Optional[str] = Field(default=None, description="Column name to use for the Y axis / values.")
    color: Optional[str] = Field(
        default=None, description="Optional column name to group/color series by."
    )
    agg: Optional[AggFunction] = Field(
        default="none", description="Aggregation to apply if the data needs grouping first."
    )
    rationale: Optional[str] = Field(default=None, description="Why this chart type was chosen.")

    @model_validator(mode="before")
    @classmethod
    def normalize_model_synonyms(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if not data.get("title"):
            data["title"] = "Analysis result"
        if "agg" not in data and "aggregation" in data:
            data["agg"] = data["aggregation"]
        if data.get("agg") is None:
            data["agg"] = "none"
        if isinstance(data.get("chart_type"), str):
            data["chart_type"] = data["chart_type"].strip().lower()
        if isinstance(data.get("agg"), str):
            data["agg"] = data["agg"].strip().lower()
        return data


class InteractiveVisualizationPlan(BaseModel):
    """LLM plan for a user-described chart over an existing result set."""

    chart_type: InteractiveChartType
    title: str = Field(description="Short chart title reflecting the user's request.")
    x: Optional[str] = Field(
        default=None,
        description="Exact retrieved column name for the X axis.",
    )
    y: Optional[str] = Field(
        default=None,
        description="Exact numeric retrieved column name for the Y axis, or null for row counts.",
    )
    color: Optional[str] = Field(
        default=None,
        description="Exact retrieved column name used to split series, or null.",
    )
    aggregation: AggFunction = "none"
    time_grain: TimeGrain = "none"
    explanation: str = Field(
        description="One sentence explaining how the requested graph will be constructed."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_model_synonyms(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if not data.get("title"):
            data["title"] = "Retrieved data"
        if "aggregation" not in data and "agg" in data:
            data["aggregation"] = data["agg"]
        if data.get("aggregation") is None:
            data["aggregation"] = "none"
        if data.get("time_grain") is None:
            data["time_grain"] = "none"
        if not data.get("explanation"):
            data["explanation"] = (
                data.get("rationale")
                or data.get("description")
                or "Uses the selected retrieved-data columns."
            )
        for field_name in ("chart_type", "aggregation", "time_grain"):
            if isinstance(data.get(field_name), str):
                data[field_name] = data[field_name].strip().lower()
        return data


class InsightResult(BaseModel):
    """Output of the result-analysis node."""

    summary: str = Field(description="A concise (2-4 sentence) natural-language insight.")
    key_findings: list[str] = Field(
        default_factory=list, description="Optional short bullet points of notable findings."
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_model_synonyms(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if not data.get("summary"):
            data["summary"] = (
                data.get("insight")
                or data.get("analysis")
                or data.get("narrative")
            )
        if "key_findings" not in data:
            findings = (
                data.get("findings")
                or data.get("key_points")
                or data.get("highlights")
                or []
            )
            data["key_findings"] = [findings] if isinstance(findings, str) else findings
        return data


class ResultPresentation(BaseModel):
    """One LLM response containing both narrative and chart recommendation."""

    insight: InsightResult
    visualization: VisualizationPlan

    @model_validator(mode="before")
    @classmethod
    def normalize_flat_response(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        if "insight" not in data and any(
            key in data for key in ("summary", "key_findings", "findings")
        ):
            data["insight"] = {
                "summary": data.get("summary"),
                "key_findings": data.get("key_findings", data.get("findings", [])),
            }
        if "visualization" not in data and "chart_type" in data:
            data["visualization"] = {
                key: data.get(key)
                for key in ("chart_type", "title", "x", "y", "color", "agg", "rationale")
            }
        return data


class MetadataEnrichmentResult(BaseModel):
    """Output of LLM-assisted metadata enrichment for a newly-discovered table."""

    table_description: str = Field(description="One-sentence business description of the table.")
    column_descriptions: dict[str, str] = Field(
        default_factory=dict,
        description="Map of column name -> one-sentence business description.",
    )


class MetadataBatchEnrichmentResult(BaseModel):
    """Descriptions for several newly discovered tables in one LLM response."""

    tables: dict[str, MetadataEnrichmentResult] = Field(default_factory=dict)
