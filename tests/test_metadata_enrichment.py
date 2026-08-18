import sqlite3

from app.llm.schemas import MetadataBatchEnrichmentResult, MetadataEnrichmentResult
from app.metadata import enrichment, store


class BatchLLM:
    def __init__(self):
        self.calls = 0

    def complete_json(self, *, system_prompt, user_prompt, schema, **_kwargs):
        self.calls += 1
        assert schema is MetadataBatchEnrichmentResult
        assert "alpha" in user_prompt
        assert "beta" in user_prompt
        return MetadataBatchEnrichmentResult(
            tables={
                "alpha": MetadataEnrichmentResult(
                    table_description="Alpha records.",
                    column_descriptions={"alpha_id": "Alpha identifier."},
                ),
                "beta": MetadataEnrichmentResult(
                    table_description="Beta records.",
                    column_descriptions={"beta_id": "Beta identifier."},
                ),
            }
        )


def test_new_table_descriptions_are_batched_into_one_llm_request():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        "CREATE TABLE alpha (alpha_id INTEGER PRIMARY KEY);"
        "CREATE TABLE beta (beta_id INTEGER PRIMARY KEY);"
    )
    llm = BatchLLM()
    enrich_fn = enrichment.make_llm_enrich_fn(llm)

    metadata, _context = store.build_metadata(
        conn,
        semantic_ctx={"tables": {}, "glossary": {}},
        enrich_fn=enrich_fn,
    )

    assert llm.calls == 1
    assert metadata["tables"]["alpha"]["description"] == "Alpha records."
    assert metadata["tables"]["beta"]["description"] == "Beta records."
