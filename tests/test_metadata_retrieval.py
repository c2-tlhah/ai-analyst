from app.metadata import retrieval
from app.metadata.retrieval import (
    MAX_RELEVANT_TABLES,
    select_relevant_tables,
    select_relevant_tables_rag,
)


def _metadata(table_count=12, relationships=None):
    tables = {
        f"FactChannel{index}": {
            "description": "Sales facts for one channel.",
            "columns": {
                "ChannelSalesAmount": {
                    "description": "Revenue for this channel.",
                    "sample_values": [],
                }
            },
        }
        for index in range(table_count)
    }
    return {"tables": tables, "relationships": relationships or []}


def test_retrieval_context_has_a_table_budget():
    selected = select_relevant_tables(_metadata(), "sales revenue")
    assert len(selected) == MAX_RELEVANT_TABLES


def test_table_hints_are_case_insensitive():
    selected = select_relevant_tables(
        _metadata(2), "unmatched wording", hinted_tables=["factchannel1"]
    )
    assert "FactChannel1" in selected


def test_camel_case_identifiers_match_natural_language():
    selected = select_relevant_tables(_metadata(2), "channel sales amount")
    assert selected


def test_rag_uses_vector_matches_ranked_first(monkeypatch):
    monkeypatch.setattr(
        retrieval.vector_store,
        "query_relevant_tables",
        lambda question, db_identity, top_k: [("FactChannel1", 0.1), ("FactChannel0", 0.2)],
    )
    selected, mode = select_relevant_tables_rag(
        _metadata(2), "wording with no lexical overlap at all", db_identity="db123"
    )
    assert mode == "vector"
    assert selected == ["FactChannel1", "FactChannel0"]


def test_rag_falls_back_to_lexical_when_vector_store_empty(monkeypatch):
    monkeypatch.setattr(
        retrieval.vector_store,
        "query_relevant_tables",
        lambda question, db_identity, top_k: None,
    )
    selected, mode = select_relevant_tables_rag(
        _metadata(2), "channel sales amount", db_identity="db123"
    )
    assert mode == "lexical"
    assert selected == select_relevant_tables(_metadata(2), "channel sales amount")


def test_rag_skips_vector_lookup_without_db_identity(monkeypatch):
    def _fail(*args, **kwargs):
        raise AssertionError("vector store should not be queried without a db_identity")

    monkeypatch.setattr(retrieval.vector_store, "query_relevant_tables", _fail)
    selected, mode = select_relevant_tables_rag(_metadata(2), "channel sales amount")
    assert mode == "lexical"
    assert selected


def test_rag_expands_vector_matches_with_foreign_key_neighbors(monkeypatch):
    metadata = _metadata(
        2,
        relationships=[
            {
                "from_table": "FactChannel0",
                "from_column": "ChannelKey",
                "to_table": "FactChannel1",
                "to_column": "ChannelKey",
            }
        ],
    )
    monkeypatch.setattr(
        retrieval.vector_store,
        "query_relevant_tables",
        lambda question, db_identity, top_k: [("FactChannel0", 0.1)],
    )
    selected, mode = select_relevant_tables_rag(metadata, "irrelevant wording", db_identity="db123")
    assert mode == "vector"
    assert set(selected) == {"FactChannel0", "FactChannel1"}


def test_rag_unions_hinted_tables_into_vector_selection(monkeypatch):
    monkeypatch.setattr(
        retrieval.vector_store,
        "query_relevant_tables",
        lambda question, db_identity, top_k: [("FactChannel0", 0.1)],
    )
    selected, mode = select_relevant_tables_rag(
        _metadata(3), "irrelevant wording", hinted_tables=["factchannel2"], db_identity="db123"
    )
    assert mode == "vector"
    assert set(selected) == {"FactChannel0", "FactChannel2"}
