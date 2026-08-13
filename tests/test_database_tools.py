"""Contract and safety tests for the audited database tool registry."""

from __future__ import annotations

import pytest

from app.metadata import store
from app.tools.database import DatabaseToolError, call_database_tool


def _metadata() -> dict:
    metadata = store.load_schema_metadata()
    assert metadata is not None
    return metadata


def test_catalog_schema_relationship_and_profile_tools_are_audited():
    combined = call_database_tool("inspect_database_schema", stage="test")
    assert "DimProduct" in combined.value["tables"]
    assert combined.value["schemas"]["DimProduct"]["columns"]
    assert combined.value["profiles"]["DimProduct"]["columns"]
    assert combined.value["relationships"]
    assert "in one call" in combined.record.summary

    tables = call_database_tool("list_tables", stage="test")
    assert "DimProduct" in tables.value
    assert tables.record.status == "completed"
    assert tables.record.name == "list_tables"

    schema = call_database_tool(
        "get_table_schema", {"table": "DimProduct"}, stage="test"
    )
    assert schema.value["name"] == "DimProduct"
    assert schema.value["columns"]

    profile = call_database_tool(
        "profile_columns", {"table": "DimProduct"}, stage="test"
    )
    assert profile.value["table"] == "DimProduct"
    assert len(profile.value["columns"]) == len(schema.value["columns"])

    relationships = call_database_tool("get_relationships", stage="test")
    assert relationships.value
    assert all("from_table" in relation for relation in relationships.value)


def test_unknown_tools_and_tables_are_blocked_with_failed_audit_records():
    with pytest.raises(DatabaseToolError) as unknown_tool:
        call_database_tool("drop_everything", stage="test")
    assert unknown_tool.value.tool_record.status == "failed"
    assert unknown_tool.value.tool_record.name == "drop_everything"

    with pytest.raises(DatabaseToolError) as unknown_table:
        call_database_tool(
            "get_table_schema",
            {"table": 'DimProduct"; DROP TABLE DimProduct;--'},
            stage="test",
        )
    assert unknown_table.value.tool_record.status == "failed"
    assert "Unknown or unavailable table" in str(unknown_table.value)


def test_query_tools_search_validate_and_execute_read_only():
    metadata = _metadata()
    search = call_database_tool(
        "search_schema",
        {"question": "count products", "hinted_tables": ["DimProduct"]},
        stage="test",
        metadata=metadata,
    )
    assert "DimProduct" in search.value["metadata"]["tables"]
    assert search.value["mode"] in {"vector", "lexical"}

    validation = call_database_tool(
        "validate_sql",
        {"sql": "SELECT COUNT(*) AS product_count FROM DimProduct"},
        stage="test",
        metadata=search.value["metadata"],
    )
    assert validation.value.is_valid is True

    execution = call_database_tool(
        "execute_readonly_sql",
        {"sql": validation.value.sanitized_sql},
        stage="test",
    )
    assert execution.value.success is True
    assert execution.value.row_count == 1
    assert execution.record.summary.startswith("Read-only query returned")

    rejected = call_database_tool(
        "validate_sql",
        {"sql": "DELETE FROM DimProduct"},
        stage="test",
        metadata=metadata,
    )
    assert rejected.value.is_valid is False
    assert rejected.record.status == "completed"
    assert "validation found" in rejected.record.summary
