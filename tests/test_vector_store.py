"""Tests for the local, versioned Chroma vector knowledge base (RAG schema
retrieval).

Runs against a real Chroma client backed by a temp directory -- not mocked --
since the point is verifying actual embedding/similarity/versioning
behavior. Chroma's bundled embedding model runs fully on-machine (no
per-query network call), so this needs no credentials, only the one-time
ONNX model download the first time it runs in a given environment.
"""

from __future__ import annotations

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.metadata import vector_store


def _fresh_client(tmp_path):
    return chromadb.PersistentClient(
        path=str(tmp_path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def _patch(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_store, "_get_client", lambda: _fresh_client(tmp_path / "chroma"))
    monkeypatch.setattr(vector_store, "_vector_directory", lambda: tmp_path)


def _metadata(schema_hash="hash-v1", extra_column=False):
    sales_columns = {
        "SalesAmount": {
            "sql_type": "REAL",
            "semantic_role": "measure",
            "description": "Revenue for the order line.",
            "sample_values": [],
        },
    }
    if extra_column:
        sales_columns["DiscountAmount"] = {
            "sql_type": "REAL",
            "semantic_role": "measure",
            "description": "Discount applied to the order line.",
            "sample_values": [],
        }
    return {
        "schema_hash": schema_hash,
        "tables": {
            "FactInternetSales": {
                "kind": "fact",
                "description": "Direct consumer sales transactions with revenue amounts.",
                "row_count": 1000,
                "primary_key": ["SalesOrderNumber"],
                "columns": sales_columns,
            },
            "DimProduct": {
                "kind": "dimension",
                "description": "Product catalog with names and categories.",
                "row_count": 50,
                "primary_key": ["ProductKey"],
                "columns": {
                    "EnglishProductName": {
                        "sql_type": "TEXT",
                        "semantic_role": "categorical_attribute",
                        "description": "The product's display name.",
                        "sample_values": ["Road Bike", "Mountain Bike"],
                    },
                },
            },
        },
    }


def test_sync_and_query_finds_semantically_relevant_table(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)

    version = vector_store.sync_collection(_metadata(), db_identity="unit_test_db")
    assert version["version"] == 1
    assert version["tables"] == ["DimProduct", "FactInternetSales"]

    matches = vector_store.query_relevant_tables(
        "how much revenue did we make", db_identity="unit_test_db", top_k=2
    )
    assert matches is not None
    assert matches[0][0] == "FactInternetSales"


def test_query_returns_none_for_unindexed_database(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    matches = vector_store.query_relevant_tables(
        "anything", db_identity="never_indexed", top_k=5
    )
    assert matches is None


def test_sync_collection_with_no_tables_is_a_noop(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    assert vector_store.sync_collection({"tables": {}}, db_identity="empty_db") is None


def test_collection_stats_before_any_sync_reports_not_indexed(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    stats = vector_store.collection_stats("brand_new_db")
    assert stats == {"indexed": False, "table_count": 0, "version": 0, "version_count": 0}


def test_unchanged_schema_reuses_the_same_version(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)

    first = vector_store.sync_collection(_metadata(schema_hash="h1"), db_identity="db1")
    second = vector_store.sync_collection(_metadata(schema_hash="h1"), db_identity="db1")

    assert first["version"] == 1
    assert second["version"] == 1
    assert vector_store.collection_stats("db1")["version_count"] == 1


def test_schema_change_mints_a_new_version_and_preserves_the_old_one(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)

    vector_store.sync_collection(_metadata(schema_hash="h1"), db_identity="db1")
    vector_store.sync_collection(
        _metadata(schema_hash="h2", extra_column=True), db_identity="db1"
    )

    versions = vector_store.list_versions("db1")
    assert [v["version"] for v in versions] == [1, 2]

    stats = vector_store.collection_stats("db1")
    assert stats["version"] == 2
    assert stats["version_count"] == 2

    # The old version's collection/documents are still there, untouched --
    # retrieval that pinned to it (or a user browsing history) still works.
    v1_docs = vector_store.read_version_documents("db1", 1)
    v2_docs = vector_store.read_version_documents("db1", 2)
    assert "DiscountAmount" not in v1_docs["FactInternetSales"]
    assert "DiscountAmount" in v2_docs["FactInternetSales"]

    # Retrieval always targets the latest version.
    matches = vector_store.query_relevant_tables("revenue", db_identity="db1", top_k=2)
    latest_collection = vector_store._get_client().get_collection(versions[-1]["collection"])
    assert {m[0] for m in matches} <= set(latest_collection.get(include=[])["ids"])


def test_txt_export_written_for_every_table(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    vector_store.sync_collection(_metadata(), db_identity="db1")

    docs = vector_store.read_version_documents("db1", 1)
    assert set(docs) == {"FactInternetSales", "DimProduct"}
    assert "Direct consumer sales transactions" in docs["FactInternetSales"]

    txt_dir = tmp_path / "knowledge_base_txt" / "db1" / "v1"
    assert (txt_dir / "FactInternetSales.txt").exists()
    assert (txt_dir / "DimProduct.txt").exists()
    assert (txt_dir / "_all_tables.txt").exists()


def test_read_version_documents_for_missing_version_is_empty(tmp_path, monkeypatch):
    _patch(monkeypatch, tmp_path)
    assert vector_store.read_version_documents("never_indexed", 1) == {}
