"""Tests for the orchestrator's session-memory/caching layer.

Covers the two optimizations added on top of the base pipeline: the
exact-question answer cache (repeat questions skip the LLM entirely) and
the metadata TTL cache (schema is verified once per session, not once per
question).
"""

import pytest

from app import orchestrator
from tests.fakes import FakeLLMClient

GOOD_SQL = (
    "SELECT p.ProductLine, SUM(f.SalesAmount) AS total_sales "
    "FROM FactInternetSales f JOIN DimProduct p ON f.ProductKey = p.ProductKey "
    "GROUP BY p.ProductLine ORDER BY total_sales DESC"
)


@pytest.fixture(autouse=True)
def _clean_orchestrator_caches():
    orchestrator.clear_session_caches()
    orchestrator._metadata_cache["metadata"] = None
    orchestrator._metadata_cache["checked_at"] = 0.0
    orchestrator._metadata_cache["source"] = None
    yield
    orchestrator.clear_session_caches()


def test_repeat_question_is_served_from_cache_without_new_llm_calls():
    llm = FakeLLMClient(sql=GOOD_SQL, relevant_tables=["DimProduct", "FactInternetSales"])

    first = orchestrator.answer_question("total sales by product line", llm_client=llm)
    assert first.status == "ok"
    assert first.cache_hit is False
    assert llm.sql_call_count == 1

    second = orchestrator.answer_question("total sales by product line", llm_client=llm)
    assert second.status == "ok"
    assert second.cache_hit is True
    assert second.elapsed_seconds == 0.0
    # No new SQL-generation call was made -- served entirely from cache.
    assert llm.sql_call_count == 1
    assert second.dataframe is not None


def test_cache_key_is_case_and_whitespace_insensitive():
    llm = FakeLLMClient(sql=GOOD_SQL, relevant_tables=["DimProduct", "FactInternetSales"])

    orchestrator.answer_question("Total sales by product line", llm_client=llm)
    second = orchestrator.answer_question("  total   SALES by product line  ", llm_client=llm)

    assert second.cache_hit is True
    assert llm.sql_call_count == 1


def test_answers_are_cached_separately_for_each_llm_client():
    first_llm = FakeLLMClient(
        sql=GOOD_SQL, relevant_tables=["DimProduct", "FactInternetSales"]
    )
    second_llm = FakeLLMClient(
        sql=GOOD_SQL, relevant_tables=["DimProduct", "FactInternetSales"]
    )

    orchestrator.answer_question("total sales by product line", llm_client=first_llm)
    second = orchestrator.answer_question(
        "total sales by product line", llm_client=second_llm
    )

    assert second.cache_hit is False
    assert first_llm.sql_call_count == 1
    assert second_llm.sql_call_count == 1


def test_use_cache_false_bypasses_cache():
    llm = FakeLLMClient(sql=GOOD_SQL, relevant_tables=["DimProduct", "FactInternetSales"])

    orchestrator.answer_question("total sales by product line", llm_client=llm)
    second = orchestrator.answer_question(
        "total sales by product line", llm_client=llm, use_cache=False
    )

    assert second.cache_hit is False
    assert llm.sql_call_count == 2


def test_failed_answers_are_not_cached():
    llm = FakeLLMClient(sql=GOOD_SQL, always_fail=True)

    first = orchestrator.answer_question("total sales by product line", llm_client=llm)
    assert first.status == "error"
    assert first.error_title == "A safe read-only query could not be generated"
    assert first.error_stage == "validation"
    assert first.error_suggestions

    second = orchestrator.answer_question("total sales by product line", llm_client=llm)
    assert second.cache_hit is False  # retried for real, not served stale from cache


def test_metadata_is_verified_once_and_then_served_from_session_cache():
    llm = FakeLLMClient(sql=GOOD_SQL, relevant_tables=["DimProduct", "FactInternetSales"])

    orchestrator.answer_question("total sales by product line", llm_client=llm)
    info_after_first = orchestrator.get_metadata_cache_info()
    assert info_after_first["cached"] is True
    assert info_after_first["source"] in {"verified", "rebuilt"}

    orchestrator.answer_question("a totally different question about resellers", llm_client=llm)
    info_after_second = orchestrator.get_metadata_cache_info()
    assert info_after_second["source"] == "session_cache"


def test_download_contains_rows_beyond_the_analysis_window():
    llm = FakeLLMClient(
        sql=(
            "SELECT ProductKey, SalesOrderNumber, SalesAmount "
            "FROM FactInternetSales ORDER BY SalesOrderNumber"
        ),
        relevant_tables=["FactInternetSales"],
    )

    response = orchestrator.answer_question(
        "Show every internet sales transaction for export",
        llm_client=llm,
        use_cache=False,
    )

    assert response.status == "ok"
    assert response.dataframe is not None
    assert response.download_dataframe is not None
    assert len(response.dataframe) == 5000
    assert response.download_row_count > len(response.dataframe)
    assert len(response.download_dataframe) == response.download_row_count
    assert response.download_truncated is False
