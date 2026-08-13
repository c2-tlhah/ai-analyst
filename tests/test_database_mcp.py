"""Protocol-level tests for the read-only database MCP gateway."""

from app.mcp_client.database import DATABASE_MCP_TOOLS, call_database_mcp_tool


def test_database_mcp_exposes_only_bounded_readonly_and_rag_tools():
    assert DATABASE_MCP_TOOLS == {
        "search_schema",
        "search_knowledge_documents",
        "resolve_relative_time",
        "validate_readonly_sql",
        "execute_readonly_sql",
        "get_database_rag_status",
    }
    assert not any(
        word in name
        for name in DATABASE_MCP_TOOLS
        for word in ("delete", "drop", "update", "write")
    )


def test_data_query_round_trip_uses_mcp_and_rag():
    search = call_database_mcp_tool(
        "search_schema",
        {"question": "Count products in DimProduct", "top_k": 3},
        stage="test_schema_rag",
    )
    assert list(search.value["metadata"]["tables"]) == ["DimProduct"]
    assert search.record.transport == "mcp"
    assert "MCP + RAG" in search.record.summary

    validation = call_database_mcp_tool(
        "validate_readonly_sql",
        {"sql": "SELECT COUNT(*) AS product_count FROM DimProduct"},
        stage="test_validation",
    )
    assert validation.value.is_valid is True
    assert validation.record.transport == "mcp"

    execution = call_database_mcp_tool(
        "execute_readonly_sql",
        {"sql": validation.value.sanitized_sql},
        stage="test_execution",
    )
    assert execution.value.success is True
    assert execution.value.row_count == 1
    assert execution.value.dataframe.iloc[0]["product_count"] > 0
    assert execution.record.transport == "mcp"


def test_knowledge_documents_are_retrieved_through_mcp_rag():
    retrieval = call_database_mcp_tool(
        "search_knowledge_documents",
        {"question": "What does StandardCost mean?", "top_k": 2},
        stage="test_knowledge_rag",
    )
    assert retrieval.value["mode"] in {"vector", "lexical"}
    assert retrieval.value["documents"]
    assert retrieval.record.transport == "mcp"
    assert "MCP + RAG" in retrieval.record.summary


def test_relative_time_is_resolved_once_for_all_fact_tables():
    result = call_database_mcp_tool(
        "resolve_relative_time",
        {
            "question": "Which product sold most last year?",
            "table_names": [
                "DimProduct",
                "FactInternetSales",
                "FactResellerSales",
            ],
        },
        stage="test_time_resolution",
    )

    assert result.value["target_year"] == 2013
    assert result.value["start_date"] == "2013-01-01"
    assert result.value["end_date_exclusive"] == "2014-01-01"
    assert set(result.value["table_date_columns"]) == {
        "FactInternetSales",
        "FactResellerSales",
    }
    assert result.record.transport == "mcp"
