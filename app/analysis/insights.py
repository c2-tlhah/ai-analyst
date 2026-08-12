"""Result analysis: turn a query result DataFrame into a natural-language insight.

All numeric summarization (describe/aggregates) happens in Pandas, deterministically,
before anything is sent to the LLM. The LLM only ever sees a compact statistical
summary -- never the raw row-by-row data beyond a small preview -- and returns a
short structured insight, never code.
"""

from __future__ import annotations

import json

import pandas as pd

from app.llm.client import LLMClient
from app.llm.schemas import InsightResult

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

MAX_PREVIEW_ROWS = 10


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
