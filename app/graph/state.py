"""LangGraph state definition for the analytics workflow.

A single ``TypedDict`` flows through every node; each node returns only the
keys it changes and LangGraph merges them into the running state. Nothing
in here is serialized across process boundaries, so it's fine to carry a
live Pandas DataFrame / Plotly Figure directly.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

import pandas as pd
import plotly.graph_objects as go


class AnalystState(TypedDict, total=False):
    # Input
    question: str
    metadata: dict[str, Any]
    # Recent (question, sql) pairs from this session, most recent last --
    # bounded, backend-managed "memory" for resolving follow-up questions.
    conversation_history: list[dict[str, Any]]

    # Intent understanding
    intent: Optional[dict[str, Any]]

    # Metadata retrieval
    relevant_metadata: Optional[dict[str, Any]]
    metadata_text: Optional[str]

    # SQL generation
    sql: Optional[str]
    sql_explanation: Optional[str]

    # SQL validation
    validation_errors: list[str]
    sanitized_sql: Optional[str]
    download_sql: Optional[str]

    # SQL execution
    execution_error: Optional[str]
    dataframe: Optional[pd.DataFrame]
    row_count: int
    truncated: bool

    # Retry control
    retry_count: int
    max_retries: int

    # Result analysis
    insight: Optional[dict[str, Any]]

    # Visualization
    viz_plan: Optional[dict[str, Any]]
    chart: Optional[go.Figure]

    # Final output
    status: str  # "ok" | "error"
    final_response: Optional[dict[str, Any]]
