"""Tests for direct LLM question answering over retrieved knowledge documents."""

from __future__ import annotations

import pytest

from app import orchestrator
from app.metadata.vector_store import RetrievedDocument
from tests.fakes import FakeLLMClient


@pytest.fixture(autouse=True)
def _clear_caches():
    orchestrator.clear_session_caches()
    yield
    orchestrator.clear_session_caches()


def _ready_status(_identity: str) -> dict:
    return {
        "status": "ready",
        "version": 2,
        "indexed": True,
        "document_count": 1,
    }


def _revenue_document(*_args, **_kwargs):
    return [
        RetrievedDocument(
            table_name="FactInternetSales",
            content=(
                "Table FactInternetSales\n"
                "SalesAmount: Net revenue recognized for this sales line.\n"
                "Aggregation guidance: SalesAmount uses sum."
            ),
            distance=0.12,
            version=2,
        )
    ]


def test_knowledge_question_is_grounded_in_retrieved_documents(monkeypatch):
    monkeypatch.setattr(orchestrator.vector_store, "collection_stats", _ready_status)
    monkeypatch.setattr(
        orchestrator.vector_store,
        "query_relevant_documents",
        _revenue_document,
    )
    llm = FakeLLMClient(
        sql="SELECT 1",
        text_response=(
            "Revenue is the sum of SalesAmount, the net revenue for each sales line "
            "[FactInternetSales]."
        ),
    )

    response = orchestrator.answer_knowledge_question(
        "What does revenue mean?",
        llm_client=llm,
        use_cache=False,
    )

    assert response.status == "ok"
    assert "[FactInternetSales]" in response.answer
    assert [source.table_name for source in response.sources] == ["FactInternetSales"]
    assert response.sources[0].version == 2
    assert llm.sql_call_count == 0
    assert len(llm.text_calls) == 1
    assert "Use only the supplied sources" in llm.text_calls[0]["system_prompt"]
    assert "SalesAmount: Net revenue" in llm.text_calls[0]["user_prompt"]


def test_repeat_knowledge_question_uses_versioned_cache(monkeypatch):
    monkeypatch.setattr(orchestrator.vector_store, "collection_stats", _ready_status)
    monkeypatch.setattr(
        orchestrator.vector_store,
        "query_relevant_documents",
        _revenue_document,
    )
    llm = FakeLLMClient(sql="SELECT 1", text_response="Revenue definition.")

    first = orchestrator.answer_knowledge_question("Define revenue", llm_client=llm)
    second = orchestrator.answer_knowledge_question("  define REVENUE ", llm_client=llm)

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert second.elapsed_seconds == 0.0
    assert len(llm.text_calls) == 1


def test_knowledge_question_explains_when_index_is_not_built(monkeypatch):
    monkeypatch.setattr(
        orchestrator.vector_store,
        "collection_stats",
        lambda _identity: {"status": "not_built", "version": 0},
    )
    llm = FakeLLMClient(sql="SELECT 1")

    response = orchestrator.answer_knowledge_question(
        "What is SalesAmount?",
        llm_client=llm,
    )

    assert response.status == "error"
    assert response.error_title == "The knowledge base is not ready"
    assert "No indexed knowledge documents" in response.error
    assert response.error_suggestions
    assert llm.text_calls == []


def test_empty_knowledge_question_is_rejected_before_retrieval():
    response = orchestrator.answer_knowledge_question("   ")

    assert response.status == "error"
    assert response.error_title == "Enter a knowledge-base question"

