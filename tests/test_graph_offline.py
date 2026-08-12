"""End-to-end LangGraph pipeline tests using a fake LLM client.

These exercise the full node wiring (intent -> metadata -> generate SQL ->
validate -> execute -> analyze -> plan viz -> chart -> respond), including
the bounded correction loop, without ever calling out to Azure.
"""

from app.db.connection import readonly_connection
from app.graph.workflow import _configure_langchain_runtime, build_workflow
from app.metadata import store
from tests.fakes import FakeLLMClient

GOOD_SQL = (
    "SELECT p.ProductLine, SUM(f.SalesAmount) AS total_sales "
    "FROM FactInternetSales f JOIN DimProduct p ON f.ProductKey = p.ProductKey "
    "GROUP BY p.ProductLine ORDER BY total_sales DESC"
)


def test_langchain_runtime_compatibility_globals(monkeypatch):
    """New LangChain roots must still work with LangChain Core 0.3 callbacks."""
    from types import SimpleNamespace

    import langchain_core.globals as langchain_globals

    fake_langchain = SimpleNamespace()
    monkeypatch.setattr(langchain_globals, "_HAS_LANGCHAIN", True)
    monkeypatch.setattr(langchain_globals, "langchain", fake_langchain, raising=False)

    _configure_langchain_runtime()

    assert fake_langchain.debug is False
    assert fake_langchain.verbose is False
    assert fake_langchain.llm_cache is None


def _metadata():
    with readonly_connection() as conn:
        metadata, _ = store.refresh_if_needed(conn)
    return metadata


def _initial_state(question: str, metadata: dict, max_retries: int = 2):
    return {
        "question": question,
        "metadata": metadata,
        "retry_count": 0,
        "max_retries": max_retries,
        "validation_errors": [],
    }


def test_happy_path_produces_dataframe_insight_and_chart():
    metadata = _metadata()
    llm = FakeLLMClient(sql=GOOD_SQL, relevant_tables=["DimProduct", "FactInternetSales"])
    workflow = build_workflow(llm)

    final_state = workflow.invoke(_initial_state("total sales by product line", metadata))

    assert final_state["status"] == "ok"
    response = final_state["final_response"]
    assert response["status"] == "ok"
    assert response["dataframe"] is not None
    assert len(response["dataframe"]) > 0
    assert response["insight"]["summary"]
    assert response["chart"] is not None
    assert response["retry_count"] == 0


def test_cross_channel_product_ranking_uses_valid_sqlite_template():
    metadata = _metadata()
    llm = FakeLLMClient(sql="SELECT TOP(10) broken FROM Nowhere")
    workflow = build_workflow(llm)

    final_state = workflow.invoke(
        _initial_state(
            "What are the top 10 products by total sales amount across both channels?",
            metadata,
        )
    )

    assert final_state["status"] == "ok"
    response = final_state["final_response"]
    assert response["retry_count"] == 0
    assert len(response["dataframe"]) == 10
    assert list(response["dataframe"].columns) == ["ProductName", "TotalSales"]
    assert "UNION ALL" in response["sql"]
    assert "FactInternetSales" in response["sql"]
    assert "FactResellerSales" in response["sql"]
    assert "LIMIT 10" in response["sql"]
    assert "TOP" not in response["sql"].upper()
    assert "IssueType" not in response["sql"]
    # Intent/insight/chart still use the selected LLM; only unambiguous SQL is templated.
    assert llm.sql_call_count == 0


def test_invalid_table_triggers_correction_then_succeeds():
    metadata = _metadata()
    llm = FakeLLMClient(sql=GOOD_SQL, fail_first_n_sql=1)
    workflow = build_workflow(llm)

    final_state = workflow.invoke(_initial_state("total sales by product line", metadata))

    assert final_state["status"] == "ok"
    assert final_state["final_response"]["retry_count"] == 1
    # SQLGenerationResult must have been requested twice: once bad, once corrected.
    assert llm.sql_call_count == 2


def test_persistent_failure_gives_up_after_max_retries():
    metadata = _metadata()
    llm = FakeLLMClient(sql=GOOD_SQL, always_fail=True)
    workflow = build_workflow(llm)

    final_state = workflow.invoke(_initial_state("total sales by product line", metadata, max_retries=2))

    assert final_state["status"] == "error"
    response = final_state["final_response"]
    assert response["status"] == "error"
    assert response["error"]
    # initial attempt + 2 retries = 3 SQL generation calls
    assert llm.sql_call_count == 3
