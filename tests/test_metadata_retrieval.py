from app.metadata.retrieval import MAX_RELEVANT_TABLES, select_relevant_tables


def _metadata(table_count=12):
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
    return {"tables": tables, "relationships": []}


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
