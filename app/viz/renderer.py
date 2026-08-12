"""Controlled chart rendering.

This module is the *only* place a chart is actually drawn. It takes a
sanitized :class:`~app.llm.schemas.VisualizationPlan` and a DataFrame and
dispatches to a small, fixed set of Plotly Express calls -- there is no
``eval``/``exec`` and no code path that runs LLM-authored Python.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.llm.schemas import VisualizationPlan
from app.logging_config import get_logger

logger = get_logger(__name__)

_AGG_FUNCS = {"sum": "sum", "avg": "mean", "count": "count", "min": "min", "max": "max"}
_COLOR_SEQUENCE = px.colors.qualitative.Safe  # colorblind-friendly categorical palette


def _maybe_aggregate(df: pd.DataFrame, plan: VisualizationPlan) -> pd.DataFrame:
    """Group and aggregate y by x (+ color) if the plan calls for it and rows repeat."""
    if not plan.agg or plan.agg == "none" or not plan.x or not plan.y:
        return df
    if plan.x not in df.columns or plan.y not in df.columns:
        return df

    group_cols = [plan.x]
    if plan.color and plan.color != plan.x and plan.color in df.columns:
        group_cols.append(plan.color)

    if not df.duplicated(subset=group_cols).any():
        return df

    agg_func = _AGG_FUNCS.get(plan.agg, "sum")
    return df.groupby(group_cols, as_index=False)[plan.y].agg(agg_func)


def render_chart(plan: VisualizationPlan, df: pd.DataFrame) -> go.Figure | None:
    """Render ``plan`` against ``df``. Returns ``None`` for table/none/empty results."""
    if df.empty or plan.chart_type in ("none", "table"):
        return None

    data = _maybe_aggregate(df, plan)
    common = {
        "title": plan.title or None,
        "template": "plotly_white",
        "color_discrete_sequence": _COLOR_SEQUENCE,
    }

    try:
        if plan.chart_type == "bar":
            fig = px.bar(data, x=plan.x, y=plan.y, color=plan.color, **common)
        elif plan.chart_type == "line":
            fig = px.line(data, x=plan.x, y=plan.y, color=plan.color, markers=True, **common)
        elif plan.chart_type == "area":
            fig = px.area(data, x=plan.x, y=plan.y, color=plan.color, **common)
        elif plan.chart_type == "scatter":
            fig = px.scatter(data, x=plan.x, y=plan.y, color=plan.color, **common)
        elif plan.chart_type == "pie":
            fig = px.pie(data, names=plan.x, values=plan.y, **common)
        elif plan.chart_type == "histogram":
            fig = px.histogram(data, x=plan.x, color=plan.color, **common)
        elif plan.chart_type == "box":
            fig = px.box(data, x=plan.x, y=plan.y, color=plan.color, **common)
        else:
            return None
    except Exception:  # noqa: BLE001 - never let a bad plan crash the app
        logger.exception("Chart rendering failed for plan=%s", plan)
        return None

    fig.update_layout(
        margin=dict(l=40, r=20, t=60, b=40),
        legend_title_text=plan.color or "",
    )
    return fig
