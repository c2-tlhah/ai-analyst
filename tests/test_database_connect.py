"""Tests for connecting the app to a different SQLite database at runtime.

Covers ``app.orchestrator.connect_database()``: validating/switching the
active database, crawling + describing its schema, and building the
versioned vector knowledge base so RAG retrieval has something to search on
the very next question against it -- including that a real schema change
mints a new knowledge-base version while the previous one stays intact.
"""

from __future__ import annotations

import dataclasses
import shutil
import sqlite3

import pytest

from app import config as config_module
from app import orchestrator
from app.db import connection
from app.metadata import store, vector_store
from tests.fakes import FakeLLMClient

GOOD_SQL = "SELECT COUNT(*) AS n FROM DimProduct"


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path, monkeypatch):
    """Sandbox metadata/vector storage and the active database per test.

    Without this, connect_database() would read/write the real project's
    metadata_store/ and vector_store/ directories, and would leave the
    active database pointed at a file tmp_path deletes once the test ends
    -- breaking every test module that runs afterward.
    """
    base_settings = config_module.get_settings()
    sandboxed = dataclasses.replace(
        base_settings,
        metadata=dataclasses.replace(
            base_settings.metadata, directory=tmp_path / "metadata_store"
        ),
        vector=dataclasses.replace(base_settings.vector, directory=tmp_path / "vector_store"),
    )
    monkeypatch.setattr(config_module, "_settings", sandboxed)
    # The Chroma client is a lazy module-level singleton bound to whatever
    # settings.vector.directory was current when it was first constructed --
    # drop it so it rebuilds against the sandboxed directory above.
    monkeypatch.setattr(vector_store, "_client", None)

    yield base_settings

    connection.set_active_database_path(None)
    orchestrator.clear_session_caches()
    orchestrator._metadata_cache["metadata"] = None
    orchestrator._metadata_cache["checked_at"] = 0.0


def test_connect_to_a_copy_of_the_sample_database_indexes_all_tables(tmp_path, _isolated_environment):
    real_settings = _isolated_environment
    copy_path = tmp_path / "copy.db"
    shutil.copy(real_settings.database.path, copy_path)

    result = orchestrator.connect_database(str(copy_path))

    assert result.success is True
    assert result.table_count == 3
    assert result.indexed_table_count == 3
    assert result.knowledge_base_version == 1
    assert result.document_count == 3
    assert connection.get_active_database_path() == copy_path.resolve()
    tool_names = [record.name for record in result.tool_records]
    assert tool_names[:4] == [
        "connect_database",
        "list_databases",
        "get_database_info",
        "inspect_database_schema",
    ]
    assert tool_names[-3:] == [
        "generate_descriptions",
        "generate_knowledge_documents",
        "write_knowledge_documents",
    ]
    assert all(record.status == "completed" for record in result.tool_records)

    info = orchestrator.get_active_database_info()
    assert info["vector_indexed"] is True
    assert info["vector_table_count"] == 3
    assert info["vector_version"] == 1
    assert info["vector_version_count"] == 1


def test_connect_reports_a_clear_error_for_a_missing_file(tmp_path):
    missing = tmp_path / "does-not-exist.db"
    result = orchestrator.connect_database(str(missing))

    assert result.success is False
    assert "No file found" in result.message


def test_connect_accepts_a_sqlite_connection_string(tmp_path, _isolated_environment):
    real_settings = _isolated_environment
    copy_path = tmp_path / "copy2.db"
    shutil.copy(real_settings.database.path, copy_path)

    result = orchestrator.connect_database(f"sqlite:///{copy_path}")

    assert result.success is True
    assert connection.get_active_database_path() == copy_path.resolve()


def test_connect_reports_vector_backend_failure_instead_of_calling_it_disabled(
    tmp_path, _isolated_environment, monkeypatch
):
    real_settings = _isolated_environment
    copy_path = tmp_path / "broken-vector.db"
    shutil.copy(real_settings.database.path, copy_path)

    def fail_client():
        raise ModuleNotFoundError("No module named 'chromadb'")

    monkeypatch.setattr(vector_store, "_get_client", fail_client)
    result = orchestrator.connect_database(str(copy_path))

    assert result.success is True
    assert result.knowledge_base_status == "error"
    assert result.indexed_table_count == 0
    assert "knowledge-base build failed" in result.message
    assert "chromadb" in result.message
    assert "VECTOR_RAG_ENABLED=false" not in result.message


def test_question_after_connect_is_retrieved_via_vector_rag(tmp_path, _isolated_environment):
    real_settings = _isolated_environment
    copy_path = tmp_path / "copy3.db"
    shutil.copy(real_settings.database.path, copy_path)
    orchestrator.connect_database(str(copy_path))

    llm = FakeLLMClient(sql=GOOD_SQL, relevant_tables=[])
    response = orchestrator.answer_question(
        "how many products do we have", llm_client=llm, use_cache=False
    )

    assert response.status == "ok"
    assert response.retrieval_mode == "vector"


def test_reconnecting_with_unchanged_schema_keeps_the_same_kb_version(
    tmp_path, _isolated_environment
):
    real_settings = _isolated_environment
    copy_path = tmp_path / "copy4.db"
    shutil.copy(real_settings.database.path, copy_path)

    first = orchestrator.connect_database(str(copy_path))
    second = orchestrator.connect_database(str(copy_path))

    assert first.knowledge_base_version == 1
    assert second.knowledge_base_version == 1
    assert len(orchestrator.list_knowledge_base_versions()) == 1


def test_schema_change_mints_a_new_kb_version_and_keeps_the_old_one(
    tmp_path, _isolated_environment
):
    real_settings = _isolated_environment
    copy_path = tmp_path / "copy5.db"
    shutil.copy(real_settings.database.path, copy_path)

    first = orchestrator.connect_database(str(copy_path))
    assert first.knowledge_base_version == 1

    conn = sqlite3.connect(copy_path)
    conn.execute("ALTER TABLE DimProduct ADD COLUMN NewFlag INTEGER")
    conn.commit()
    conn.close()

    second = orchestrator.connect_database(str(copy_path))
    assert second.knowledge_base_version == 2

    versions = orchestrator.list_knowledge_base_versions()
    assert [v["version"] for v in versions] == [1, 2]

    v1_docs = orchestrator.get_knowledge_base_documents(1)
    v2_docs = orchestrator.get_knowledge_base_documents(2)
    assert "NewFlag" not in v1_docs["DimProduct"]
    assert "NewFlag" in v2_docs["DimProduct"]

    # Retrieval keeps working off the latest version after the change.
    llm = FakeLLMClient(sql=GOOD_SQL, relevant_tables=[])
    response = orchestrator.answer_question(
        "how many products do we have", llm_client=llm, use_cache=False
    )
    assert response.retrieval_mode == "vector"


def _build_unseen_commerce_database(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            customer_name VARCHAR(100) NOT NULL,
            region TEXT,
            created_at DATETIME
        );
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            ordered_at TIMESTAMP NOT NULL,
            net_revenue DECIMAL(12, 2) NOT NULL,
            order_status VARCHAR(20),
            is_priority BOOLEAN
        );
        CREATE VIEW completed_orders AS
            SELECT order_id, customer_id, ordered_at, net_revenue
            FROM orders WHERE order_status = 'complete';
        INSERT INTO customers VALUES
            (1, 'Aster Labs', 'North', '2026-01-01'),
            (2, 'Beacon Works', 'South', '2026-01-02');
        INSERT INTO orders VALUES
            (101, 1, '2026-02-01', 120.50, 'complete', 1),
            (102, 1, '2026-02-10', 79.50, 'pending', 0),
            (103, 2, '2026-03-05', 300.00, 'complete', 1);
        """
    )
    conn.commit()
    conn.close()


def test_unseen_database_is_discovered_documented_and_queried_end_to_end(tmp_path):
    database = tmp_path / "unseen-commerce.db"
    _build_unseen_commerce_database(database)

    connected = orchestrator.connect_database(str(database))
    assert connected.success is True
    assert connected.table_count == 3
    assert connected.document_count == 3

    metadata = store.load_schema_metadata()
    assert set(metadata["tables"]) == {"customers", "orders", "completed_orders"}
    assert metadata["glossary"] == {}
    assert "DimProduct" not in metadata["tables"]
    assert metadata["tables"]["orders"]["columns"]["net_revenue"]["semantic_role"] == "measure"
    assert metadata["tables"]["orders"]["columns"]["ordered_at"]["semantic_role"] == "temporal"
    assert metadata["tables"]["orders"]["columns"]["is_priority"]["semantic_role"] == "flag"
    assert any(
        relationship["from_table"] == "orders"
        and relationship["from_column"] == "customer_id"
        and relationship["to_table"] == "customers"
        and relationship["to_column"] == "customer_id"
        and relationship["source"] == "inferred"
        and relationship["confidence"] == 0.8
        for relationship in metadata["relationships"]
    )

    schema_path, context_path = store.metadata_paths()
    assert schema_path.exists()
    assert context_path.exists()
    assert schema_path.parent.name == connection.get_active_database_identity()

    examples = orchestrator.get_example_questions()
    assert examples
    assert any("orders" in question for question in examples)
    assert not any("FactInternetSales" in question for question in examples)

    llm = FakeLLMClient(
        sql=(
            "SELECT c.region, SUM(o.net_revenue) AS total_revenue "
            "FROM orders AS o JOIN customers AS c "
            "ON o.customer_id = c.customer_id "
            "GROUP BY c.region ORDER BY total_revenue DESC"
        ),
        relevant_tables=["orders", "customers"],
        text_response="Net revenue is documented on the orders table [orders].",
    )
    response = orchestrator.answer_question(
        "What is total net revenue by customer region?",
        llm_client=llm,
        use_cache=False,
    )
    assert response.status == "ok"
    assert list(response.dataframe.columns) == ["region", "total_revenue"]
    assert response.dataframe["total_revenue"].sum() == pytest.approx(500.0)
    assert response.download_dataframe is None
    download = orchestrator.prepare_complete_download(response)
    assert download.status == "ok"
    assert download.row_count == 2
    assert download.csv_data is not None
    assert [record.name for record in response.tool_records][:3] == [
        "search_schema",
        "validate_readonly_sql",
        "execute_readonly_sql",
    ]
    assert all(record.transport == "mcp" for record in response.tool_records)

    knowledge = orchestrator.answer_knowledge_question(
        "What does net revenue mean?",
        llm_client=llm,
        use_cache=False,
    )
    assert knowledge.status == "ok"
    assert any(source.table_name == "orders" for source in knowledge.sources)


def test_metadata_and_business_context_are_isolated_per_unseen_database(
    tmp_path, monkeypatch
):
    current = config_module.get_settings()
    monkeypatch.setattr(
        config_module,
        "_settings",
        dataclasses.replace(
            current,
            vector=dataclasses.replace(current.vector, enabled=False),
        ),
    )
    first = tmp_path / "first.db"
    second = tmp_path / "second.db"
    conn = sqlite3.connect(first)
    conn.execute("CREATE TABLE apples (apple_id INTEGER PRIMARY KEY, variety TEXT)")
    conn.commit()
    conn.close()
    conn = sqlite3.connect(second)
    conn.execute("CREATE TABLE planets (planet_id INTEGER PRIMARY KEY, planet_name TEXT)")
    conn.commit()
    conn.close()

    assert orchestrator.connect_database(str(first)).success
    first_paths = store.metadata_paths()
    first_metadata = store.load_schema_metadata()
    assert set(first_metadata["tables"]) == {"apples"}

    assert orchestrator.connect_database(str(second)).success
    second_paths = store.metadata_paths()
    second_metadata = store.load_schema_metadata()
    assert set(second_metadata["tables"]) == {"planets"}
    assert first_paths != second_paths

    assert orchestrator.connect_database(str(first)).success
    assert store.metadata_paths() == first_paths
    assert set(store.load_schema_metadata()["tables"]) == {"apples"}
    assert store.load_business_context()["glossary"] == {}
