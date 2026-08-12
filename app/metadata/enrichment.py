"""LLM-assisted description generation for newly-discovered schema objects.

Used only as a fallback: curated descriptions in
:mod:`app.metadata.business_context_seed` always win, and a humanized
column name is always the ultimate fallback if no LLM is configured. This
module is what lets the platform stay schema-aware when a table or column
appears that nobody has manually documented yet.
"""

from __future__ import annotations

from app.llm.client import LLMClient, LLMError
from app.llm.schemas import MetadataEnrichmentResult
from app.logging_config import get_logger
from app.metadata.store import EnrichFn

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a data analyst documenting a business analytics database. Given a "
    "table name, its role (fact or dimension), and a list of column names, write "
    "concise, factual one-sentence business descriptions. Do not invent business "
    "meaning you cannot infer from the names; keep descriptions short and neutral."
)


def make_llm_enrich_fn(llm_client: LLMClient) -> EnrichFn:
    """Build an ``enrich_fn`` for :func:`app.metadata.store.refresh_if_needed`."""

    def enrich(table_name: str, kind: str, columns: list[str]) -> dict:
        user_prompt = (
            f"Table name: {table_name}\n"
            f"Table role: {kind}\n"
            f"Columns needing descriptions: {', '.join(columns)}\n\n"
            "Return a table_description and a column_descriptions map covering "
            "every listed column."
        )
        try:
            result: MetadataEnrichmentResult = llm_client.complete_json(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=MetadataEnrichmentResult,
            )
        except LLMError:
            logger.exception("LLM metadata enrichment call failed for table %s", table_name)
            return {}

        return {"table": result.table_description, "columns": result.column_descriptions}

    return enrich
