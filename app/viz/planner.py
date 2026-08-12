"""Visualization planning: LLM proposes a plan, the backend sanitizes it.

The LLM's job ends at picking a chart type, axes, grouping column and
aggregation -- a small structured object. It never writes plotting code.
:func:`sanitize_plan` is a deterministic safety net that runs on *every*
plan (LLM-produced or the offline fallback) before it ever reaches
:mod:`app.viz.renderer`, so a hallucinated column name can't blow up the
chart step.
"""

from __future__ import annotations

import json

import pandas as pd

from app.llm.client import LLMClient
from app.llm.schemas import ChartType, VisualizationPlan

_SYSTEM_PROMPT = """You are the visualization planner of an analytics platform. You are
given the user's question and the columns/sample rows of the query result. Propose a
chart plan as structured JSON -- you do NOT write any code.

Guidance:
- chart_type must be one of: bar, line, scatter, pie, histogram, box, area, table, none.
- Use "line" or "area" for trends over a date/time column.
- Use "bar" for comparisons across categories or rankings ("top N").
- Use "pie" only for a simple part-of-whole breakdown with few (<= 8) categories.
- Use "scatter" for relationships between two numeric measures.
- Use "histogram" for the distribution of a single numeric column.
- Use "table" if the result is better read as a table (e.g. many columns, few rows,
  or no obviously plottable numeric measure).
- Use "none" only if the result is a single scalar value.
- x and y MUST be exact column names from the ones given -- never invent a name.
- Set agg to how y should be aggregated if the same x value repeats (sum/avg/count/
  min/max), or "none" if no aggregation is needed.
- Always include a non-empty title. Use the exact keys chart_type, title, x, y,
  color, agg, and rationale; do not rename or omit required keys.
"""


def _user_prompt(question: str, df: pd.DataFrame) -> str:
    info = {
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "row_count": len(df),
        "sample_rows": df.head(5).to_dict(orient="records"),
    }
    return f"QUESTION:\n{question}\n\nRESULT COLUMNS / SAMPLE (JSON):\n{json.dumps(info, default=str)}"


def plan_visualization(
    llm_client: LLMClient, *, question: str, df: pd.DataFrame
) -> VisualizationPlan:
    plan = llm_client.complete_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_user_prompt(question, df),
        schema=VisualizationPlan,
    )
    return sanitize_plan(plan, df)


_ALLOWED_CHART_TYPES: set[ChartType] = {
    "bar", "line", "scatter", "pie", "histogram", "box", "area", "table", "none",
}


def _first_of_dtype(df: pd.DataFrame, kinds: str) -> str | None:
    cols = df.select_dtypes(include=kinds).columns
    return cols[0] if len(cols) else None


def fallback_plan(df: pd.DataFrame, title: str = "Result") -> VisualizationPlan:
    """Deterministic plan used when the LLM is unavailable or its plan is unusable."""
    if df.empty:
        return VisualizationPlan(chart_type="table", title=title)
    if df.shape == (1, 1):
        return VisualizationPlan(chart_type="none", title=title)

    numeric_col = _first_of_dtype(df, "number")
    text_col = _first_of_dtype(df, "object")

    if numeric_col and text_col:
        return VisualizationPlan(
            chart_type="bar", title=title, x=text_col, y=numeric_col, agg="sum"
        )
    if numeric_col:
        return VisualizationPlan(chart_type="histogram", title=title, x=numeric_col)
    return VisualizationPlan(chart_type="table", title=title)


def sanitize_plan(plan: VisualizationPlan, df: pd.DataFrame) -> VisualizationPlan:
    """Deterministically validate/repair a plan against the actual DataFrame."""
    if df.empty:
        return VisualizationPlan(chart_type="table", title=plan.title or "Result")

    if plan.chart_type not in _ALLOWED_CHART_TYPES:
        return fallback_plan(df, plan.title or "Result")

    if plan.chart_type in ("none", "table"):
        return plan

    columns = set(df.columns)
    x = plan.x if plan.x in columns else None
    y = plan.y if plan.y in columns else None
    color = plan.color if plan.color in columns else None

    if plan.chart_type == "histogram":
        x = x or _first_of_dtype(df, "number")
        if not x:
            return fallback_plan(df, plan.title)
        return plan.model_copy(update={"x": x, "y": None, "color": color})

    if plan.chart_type in ("bar", "line", "area", "box"):
        x = x or _first_of_dtype(df, "object") or (df.columns[0] if len(df.columns) else None)
        y = y or _first_of_dtype(df, "number")
        if not x or not y:
            return fallback_plan(df, plan.title)
        return plan.model_copy(update={"x": x, "y": y, "color": color})

    if plan.chart_type == "scatter":
        numeric_cols = list(df.select_dtypes(include="number").columns)
        if not x or x not in numeric_cols:
            x = numeric_cols[0] if numeric_cols else None
        if not y or y not in numeric_cols or y == x:
            remaining = [c for c in numeric_cols if c != x]
            y = remaining[0] if remaining else None
        if not x or not y:
            return fallback_plan(df, plan.title)
        return plan.model_copy(update={"x": x, "y": y, "color": color})

    if plan.chart_type == "pie":
        names = x or _first_of_dtype(df, "object")
        values = y or _first_of_dtype(df, "number")
        if not names or not values:
            return fallback_plan(df, plan.title)
        distinct = df[names].nunique()
        if distinct > 12:
            # Too many slices to read -- a bar chart communicates this better.
            return plan.model_copy(update={"chart_type": "bar", "x": names, "y": values})
        return plan.model_copy(update={"x": names, "y": values, "color": color})

    return fallback_plan(df, plan.title)
