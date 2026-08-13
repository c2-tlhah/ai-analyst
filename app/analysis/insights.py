"""Result analysis: turn a query result DataFrame into a natural-language insight.

All numeric summarization (describe/aggregates) happens in Pandas, deterministically,
before anything is sent to the LLM. The LLM only ever sees a compact statistical
summary -- never the raw row-by-row data beyond a small preview -- and returns a
short structured insight, never code.
"""

from __future__ import annotations

import json
import math
import re

import pandas as pd

from app.llm.client import LLMClient
from app.llm.schemas import InsightResult, ResultPresentation

_SYSTEM_PROMPT = """You are a data analyst writing a concise insight for a business
dashboard. You are given the user's original question, the SQL query that answered
it, and a statistical summary of the result set (not the full data).

Write a 2-4 sentence plain-language summary of what the data shows in direct answer
to the question. Reference concrete numbers from the summary. Do not mention SQL,
tables, columns types, or the query itself -- write for a business audience. If the
result set is empty, say so plainly and suggest a possible reason (e.g. filters too
narrow). Optionally include a few short key_findings bullets for standout numbers.
The JSON object must include the exact key "summary". Put optional bullets in the
"key_findings" array; do not rename "summary" to "insight" or "analysis".
"""

# A model never needs the complete result set to explain it.  Keep the raw-row
# context deliberately tiny and combine it with deterministic numeric statistics
# computed locally.  This makes provider latency independent of query row count.
MAX_PREVIEW_ROWS = 5


def _display_value(value: object) -> str:
    if pd.isna(value):
        return "NULL"
    if isinstance(value, float):
        return f"{value:,.6g}"
    return str(value)


def deterministic_direct_answer(
    question: str, df: pd.DataFrame, *, sql: str = ""
) -> InsightResult | None:
    """Ground scalar and ranking answers directly in executed result cells.

    These are the answer shapes where a fluent model can most visibly replace a
    correct value/name with a plausible one. The model still plans presentation,
    but the displayed answer is constructed from the DataFrame itself.
    """
    if df.empty:
        return InsightResult(
            summary="No rows matched the validated query and requested filters.",
            key_findings=[],
        )
    if len(df) == 1:
        values = "; ".join(
            f"{column} = {_display_value(df.iloc[0][column])}" for column in df.columns
        )
        return InsightResult(summary=f"Result: {values}.", key_findings=[values])

    normalized = " ".join((question or "").casefold().split())
    if not re.search(
        r"\b(top|bottom|most|least|highest|lowest|largest|smallest|best|worst)\b",
        normalized,
    ):
        return None
    first = "; ".join(
        f"{column} = {_display_value(df.iloc[0][column])}" for column in df.columns
    )
    if re.search(r"\bPARTITION\s+BY\b", sql, flags=re.IGNORECASE):
        findings = [
            "; ".join(
                f"{column} = {_display_value(row[column])}" for column in df.columns
            )
            for _, row in df.head(3).iterrows()
        ]
        return InsightResult(
            summary=(
                f"The query returned the winner for {len(df)} group(s). "
                f"First result: {first}."
            ),
            key_findings=findings,
        )
    findings = []
    for position, (_, row) in enumerate(df.head(3).iterrows(), start=1):
        values = "; ".join(
            f"{column} = {_display_value(row[column])}" for column in df.columns
        )
        findings.append(f"#{position}: {values}")
    return InsightResult(
        summary=f"The first ranked result is {first}. The query returned {len(df)} ranked row(s).",
        key_findings=findings,
    )


_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?")


def _numeric_evidence(question: str, df: pd.DataFrame) -> list[float]:
    evidence = [float(len(df))]
    for match in _NUMBER_RE.findall(question or ""):
        try:
            evidence.append(float(match.replace(",", "")))
        except ValueError:
            pass
    for column in df.select_dtypes(include="number").columns:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        evidence.extend(float(value) for value in values if math.isfinite(float(value)))
        if not values.empty:
            described = values.describe()
            evidence.extend(
                float(value)
                for value in described.values
                if math.isfinite(float(value))
            )
    # Date-like/string cells legitimately contribute year/month/day numbers.
    for value in df.astype(str).head(MAX_PREVIEW_ROWS).to_numpy().ravel():
        for match in _NUMBER_RE.findall(value):
            try:
                evidence.append(float(match.replace(",", "")))
            except ValueError:
                pass
    return evidence


def _matches_evidence(value: float, evidence: list[float]) -> bool:
    return any(
        math.isclose(value, candidate, rel_tol=1e-6, abs_tol=1e-6)
        for candidate in evidence
    )


def deterministic_result_overview(df: pd.DataFrame) -> InsightResult:
    if df.empty:
        return InsightResult(
            summary="No rows matched the validated query and requested filters."
        )
    first = "; ".join(
        f"{column} = {_display_value(df.iloc[0][column])}" for column in df.columns
    )
    ranges = []
    for column in df.select_dtypes(include="number").columns[:4]:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if not values.empty:
            ranges.append(
                f"{column}: {_display_value(values.min())} to {_display_value(values.max())}"
            )
    range_text = f" Numeric ranges: {'; '.join(ranges)}." if ranges else ""
    return InsightResult(
        summary=f"The query returned {len(df)} row(s). First row: {first}.{range_text}",
        key_findings=[first],
    )


def enforce_numeric_grounding(
    insight: InsightResult, *, question: str, df: pd.DataFrame
) -> InsightResult:
    """Replace prose containing numbers unsupported by deterministic evidence."""
    evidence = _numeric_evidence(question, df)
    text = " ".join([insight.summary, *insight.key_findings])
    unsupported: list[float] = []
    for token in _NUMBER_RE.findall(text):
        try:
            value = float(token.replace(",", ""))
        except ValueError:
            continue
        if not _matches_evidence(value, evidence):
            unsupported.append(value)
    return deterministic_result_overview(df) if unsupported else insight


def summarize_dataframe(df: pd.DataFrame) -> dict:
    numeric_cols = df.select_dtypes(include="number").columns
    summary: dict = {
        "row_count": int(len(df)),
        "columns": [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns],
        "preview": df.head(MAX_PREVIEW_ROWS).to_dict(orient="records"),
    }
    if len(df) > 0 and len(numeric_cols) > 0:
        summary["numeric_summary"] = {
            col: df[col].describe().to_dict() for col in numeric_cols
        }
    return summary


def generate_insight(
    llm_client: LLMClient, *, question: str, sql: str, df: pd.DataFrame
) -> InsightResult:
    summary = summarize_dataframe(df)
    user_prompt = (
        f"QUESTION:\n{question}\n\nSQL USED:\n{sql}\n\n"
        f"RESULT SUMMARY (JSON):\n{json.dumps(summary, default=str)}"
    )
    return llm_client.complete_json(
        system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt, schema=InsightResult
    )


_PRESENTATION_SYSTEM_PROMPT = """You are the result-presentation stage of an
analytics platform. One validated read-only query has already run. Return one
JSON object containing BOTH:

1. insight: a concise business answer with summary and key_findings.
2. visualization: a safe chart plan with chart_type, title, x, y, color, agg,
   and rationale.

Use concrete values from the supplied deterministic result summary. Never
invent columns. Chart types are bar, line, scatter, pie, histogram, box, area,
table, or none. Prefer line/area for time, bar for categories/rankings, scatter
for two measures, histogram for one measure, table when no graph helps, and none
for a single scalar. Axis fields must be exact result-column names. Do not write
chart code or SQL. Include both nested objects even if the best chart is table
or none."""


def generate_result_presentation(
    llm_client: LLMClient,
    *,
    question: str,
    sql: str,
    df: pd.DataFrame,
    result_summary: dict | None = None,
) -> ResultPresentation:
    """Generate the answer narrative and recommended chart in one model call."""
    summary = result_summary if result_summary is not None else summarize_dataframe(df)
    user_prompt = (
        f"QUESTION:\n{question}\n\nSQL USED:\n{sql}\n\n"
        f"RESULT SUMMARY (JSON):\n{json.dumps(summary, default=str)}"
    )
    return llm_client.complete_json(
        system_prompt=_PRESENTATION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=ResultPresentation,
    )
