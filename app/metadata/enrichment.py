"""LLM-assisted description generation for newly-discovered schema objects.

Used only as a fallback: database-specific saved descriptions win, and a
humanized column name is the ultimate fallback if no LLM is configured. This
module lets unseen databases become schema-aware without
requiring hand-written metadata.
"""

from __future__ import annotations

import json

from app.llm.client import LLMClient, LLMError
from app.config import get_settings
from app.llm.schemas import MetadataBatchEnrichmentResult
from app.logging_config import get_logger
from app.metadata.store import EnrichFn
from app.observability import trace_span

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a data analyst documenting a business analytics database. Given a "
    "table name, its structural role, and discovered column facts, write concise, "
    "factual one-sentence descriptions. The schema may be from any domain and may "
    "not be a warehouse. Do not assume sales, products, customers, fact/dimension "
    "semantics, or business meaning not supported by the supplied facts. Never "
    "repeat representative values from sensitive fields. Keep descriptions neutral."
)


class _BatchLLMEnricher:
    """Callable compatible enricher with a multi-table fast path."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def __call__(
        self,
        table_name: str,
        kind: str,
        columns: list[str],
        schema_context: dict,
    ) -> dict:
        return self.enrich_many(
            [
                {
                    "table_name": table_name,
                    "kind": kind,
                    "columns": columns,
                    "schema_context": schema_context,
                }
            ]
        ).get(table_name, {})

    def enrich_many(self, requests: list[dict]) -> dict[str, dict]:
        """Describe a configurable group of tables per provider request."""
        batch_size = max(1, get_settings().metadata.llm_enrich_batch_size)
        enriched: dict[str, dict] = {}
        for offset in range(0, len(requests), batch_size):
            batch = requests[offset : offset + batch_size]
            compact = [
                {
                    "table_name": item["table_name"],
                    "kind": item["kind"],
                    "columns_needing_descriptions": item["columns"],
                    "schema_facts": item["schema_context"],
                }
                for item in batch
            ]
            user_prompt = (
                "Describe every table in this JSON array in one response:\n"
                f"{json.dumps(compact, default=str, separators=(',', ':'))}\n\n"
                "Return a tables map keyed by the exact table_name. Each value must "
                "contain table_description and a column_descriptions map covering "
                "every requested column. Use only the supplied schema facts."
            )
            try:
                with trace_span(
                    "enrich_metadata_batch",
                    category="agent_stage",
                    metadata={
                        "table_count": len(batch),
                        "table_names": [item["table_name"] for item in batch],
                        "provider": getattr(
                            self.llm_client, "provider_name", "custom"
                        ),
                        "model": getattr(
                            self.llm_client,
                            "model_name",
                            type(self.llm_client).__name__,
                        ),
                    },
                ):
                    result: MetadataBatchEnrichmentResult = (
                        self.llm_client.complete_json(
                            system_prompt=_SYSTEM_PROMPT,
                            user_prompt=user_prompt,
                            schema=MetadataBatchEnrichmentResult,
                        )
                    )
            except LLMError:
                logger.exception(
                    "Batched LLM metadata enrichment failed for %d table(s)",
                    len(batch),
                )
                continue
            by_casefold = {name.casefold(): value for name, value in result.tables.items()}
            for item in batch:
                table_name = item["table_name"]
                value = by_casefold.get(table_name.casefold())
                if value is not None:
                    enriched[table_name] = {
                        "table": value.table_description,
                        "columns": value.column_descriptions,
                    }
        return enriched


def make_llm_enrich_fn(llm_client: LLMClient) -> EnrichFn:
    """Build an enricher whose batch path minimizes hosted-provider calls."""
    return _BatchLLMEnricher(llm_client)  # type: ignore[return-value]
