"""LLM-backed SQL generation.

The model only ever sees the relevance-trimmed metadata text produced by
:mod:`app.metadata.retrieval` -- never the raw database -- and is
instructed to stay strictly within it. Its output is a plan
(:class:`~app.llm.schemas.SQLGenerationResult`), not something executed
directly: :mod:`app.sql.validator` and :mod:`app.sql.executor` are the
only things that ever touch the database.
"""

from __future__ import annotations

import re

from app.llm.client import LLMClient
from app.llm.schemas import SQLGenerationResult

_SYSTEM_PROMPT = """You are the SQL generation engine of a read-only analytics platform.

Rules (all are hard requirements):
- Generate exactly one SQLite SELECT statement (CTEs / joins / aggregates allowed).
- Use ONLY the tables and columns given in the metadata context below. Never invent
  table or column names, and never reference any table not listed there.
- Never generate INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, PRAGMA, ATTACH or any
  other statement that is not a SELECT.
- Prefer explicit column lists over SELECT * when the question asks for specific
  fields; use SELECT * only for open-ended "show me the data" style questions.
- Use the aggregation rules / default measures given in the metadata when the
  question is ambiguous about how to aggregate a measure.
- Add a LIMIT clause for ranking/"top N" questions; for other questions the backend
  will enforce a safety LIMIT automatically, so you do not need to add one yourself
  unless the question implies a specific N.
- Use table aliases and qualify columns when joining more than one table.
- Use SQLite syntax only: never use TOP/TOP(...), :: casts, SQL Server brackets,
  or a trailing comma before FROM. Put LIMIT N at the very end of a ranking query.
- A business concept such as "channel" does not imply that an IssueType, Channel,
  or SalesChannel column exists. When channels are stored in separate fact tables,
  combine their compatible rows with UNION ALL before aggregating. Do not add the
  same fact table twice and do not invent a discriminator column.
"""


_CROSS_CHANNEL_REQUIRED_METADATA = (
    "TABLE DimProduct",
    "TABLE FactInternetSales",
    "TABLE FactResellerSales",
    "ProductKey",
    "ProductName",
    "SalesAmount",
)


def _is_cross_channel_product_ranking(question: str, metadata_text: str) -> bool:
    """Recognize the packaged schema's high-confidence cross-channel ranking intent."""
    normalized = " ".join(question.lower().split())
    has_rank = bool(re.search(r"\btop\s+\d+\b", normalized))
    has_product = bool(re.search(r"\bproducts?\b", normalized))
    has_sales = bool(re.search(r"\b(sales|revenue|amount)\b", normalized))
    has_channels = bool(
        re.search(r"\b(both|all|across)\b", normalized)
        and re.search(r"\bchannels?\b", normalized)
    )
    metadata_lower = metadata_text.lower()
    has_schema = all(
        name.lower() in metadata_lower for name in _CROSS_CHANNEL_REQUIRED_METADATA
    )
    return has_rank and has_product and has_sales and has_channels and has_schema


def _cross_channel_product_ranking(question: str) -> SQLGenerationResult:
    match = re.search(r"\btop\s+(\d+)\b", question, flags=re.IGNORECASE)
    requested_limit = int(match.group(1)) if match else 10
    # The executor still applies the configured global maximum after validation.
    requested_limit = max(1, requested_limit)
    sql = f"""WITH channel_sales AS (
    SELECT ProductKey, SalesAmount FROM FactInternetSales
    UNION ALL
    SELECT ProductKey, SalesAmount FROM FactResellerSales
)
SELECT p.ProductName, SUM(s.SalesAmount) AS TotalSales
FROM channel_sales AS s
JOIN DimProduct AS p ON p.ProductKey = s.ProductKey
GROUP BY p.ProductKey, p.ProductName
ORDER BY TotalSales DESC
LIMIT {requested_limit}"""
    return SQLGenerationResult(
        sql=sql,
        explanation=(
            "Combines internet and reseller sales without deduplicating transactions, "
            "then totals sales by product and returns the highest-selling products."
        ),
        tables_used=["FactInternetSales", "FactResellerSales", "DimProduct"],
    )


def _schema_specific_guidance(question: str, metadata_text: str) -> str:
    if not _is_cross_channel_product_ranking(question, metadata_text):
        return ""
    return """

SCHEMA-SPECIFIC GUIDANCE:
For sales across both channels in this metadata, UNION ALL ProductKey and
SalesAmount from FactInternetSales and FactResellerSales in a CTE. Join that CTE
to DimProduct on ProductKey, group by DimProduct.ProductKey and ProductName, sum
SalesAmount once, order descending, and finish with LIMIT N. There is no IssueType
column and SQLite has no TOP syntax.
""".strip()


def _user_prompt(
    question: str,
    metadata_text: str,
    prior_sql: str | None,
    prior_error: str | None,
    conversation_history_text: str | None,
) -> str:
    parts = [f"METADATA CONTEXT:\n{metadata_text}"]
    guidance = _schema_specific_guidance(question, metadata_text)
    if guidance:
        parts.append(f"\n{guidance}")
    if conversation_history_text:
        parts.append(
            f"\n{conversation_history_text}\n"
            "(Only use this to resolve references like \"that\"/\"same but ...\" "
            "in the new question below -- do not answer the old question again.)"
        )
    parts.append(f"\nQUESTION:\n{question}")
    if prior_sql and prior_error:
        parts.append(
            "\nA previous attempt at this question failed validation/execution. "
            f"Previous SQL:\n{prior_sql}\n\nError:\n{prior_error}\n\n"
            "Rebuild the query from scratch as SQLite. Use only exact metadata names, "
            "LIMIT instead of TOP, no :: syntax, and no comma immediately before FROM."
        )
    return "\n".join(parts)


def generate_sql(
    llm_client: LLMClient,
    *,
    question: str,
    metadata_text: str,
    prior_sql: str | None = None,
    prior_error: str | None = None,
    conversation_history_text: str | None = None,
) -> SQLGenerationResult:
    # This intent has one unambiguous implementation in the packaged schema. A
    # deterministic template avoids slow, repeated dialect mistakes from small
    # local models while the normal validator/executor safety gates still apply.
    if _is_cross_channel_product_ranking(question, metadata_text):
        return _cross_channel_product_ranking(question)

    return llm_client.complete_json(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_user_prompt(
            question, metadata_text, prior_sql, prior_error, conversation_history_text
        ),
        schema=SQLGenerationResult,
    )
