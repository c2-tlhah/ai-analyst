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
from app.metadata import vector_store
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
    assert connection.get_active_database_path() == copy_path.resolve()

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
