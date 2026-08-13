"""End-to-end LangGraph pipeline tests using a fake LLM client.

These exercise the full node wiring (metadata -> generate SQL -> validate ->
execute -> combined insight/plan -> chart -> respond), including
the bounded correction loop, without ever calling out to Azure.
"""

import pytest

from app.db.connection import readonly_connection
from app.graph.workflow import _configure_langchain_runtime, build_workflow
from app.llm.client import LLMRateLimitError
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
    assert [record.name for record in response["tool_records"]] == [
        "search_schema",
        "validate_readonly_sql",
        "execute_readonly_sql",
    ]
    assert all(record.transport == "mcp" for record in response["tool_records"])
    # One call generates SQL; one combined call generates insight + chart plan.
    assert llm.calls == ["SQLGenerationResult", "ResultPresentation"]


def test_last_year_mixed_channel_sql_is_rejected_and_corrected():
    metadata = _metadata()
    wrong_sql = """WITH combined_sales AS (
      SELECT ProductKey, OrderQuantity FROM FactInternetSales
      WHERE strftime('%Y', OrderDate) = (
        SELECT strftime('%Y', MAX(OrderDate)) FROM FactInternetSales
      )
      UNION ALL
      SELECT ProductKey, OrderQuantity FROM FactResellerSales
      WHERE strftime('%Y', OrderDate) = (
        SELECT strftime('%Y', MAX(OrderDate)) FROM FactResellerSales
      )
    )
    SELECT p.ProductName, SUM(s.OrderQuantity) AS TotalUnits
    FROM combined_sales s JOIN DimProduct p ON p.ProductKey = s.ProductKey
    GROUP BY p.ProductKey, p.ProductName ORDER BY TotalUnits DESC LIMIT 1"""
    correct_sql = """WITH combined_sales AS (
      SELECT ProductKey, OrderQuantity FROM FactInternetSales
      WHERE date(OrderDate) >= '2013-01-01' AND date(OrderDate) < '2014-01-01'
      UNION ALL
      SELECT ProductKey, OrderQuantity FROM FactResellerSales
      WHERE date(OrderDate) >= '2013-01-01' AND date(OrderDate) < '2014-01-01'
    )
    SELECT p.ProductName, SUM(s.OrderQuantity) AS TotalUnits
    FROM combined_sales s JOIN DimProduct p ON p.ProductKey = s.ProductKey
    GROUP BY p.ProductKey, p.ProductName ORDER BY TotalUnits DESC LIMIT 1"""
    llm = FakeLLMClient(
        sql=correct_sql,
        relevant_tables=["DimProduct", "FactInternetSales", "FactResellerSales"],
        fail_first_n_sql=1,
        bad_sql=wrong_sql,
    )
    workflow = build_workflow(llm)

    final_state = workflow.invoke(
        _initial_state("Which is the most sold product in last year?", metadata)
    )

    response = final_state["final_response"]
    assert response["status"] == "ok"
    assert response["retry_count"] == 1
    assert response["time_context"]["target_year"] == 2013
    assert response["dataframe"].to_dict("records") == [
        {"ProductName": "Water Bottle - 30 oz.", "TotalUnits": 6416}
    ]
    tool_names = [record.name for record in response["tool_records"]]
    assert tool_names.count("resolve_relative_time") == 1
    assert tool_names.count("validate_readonly_sql") == 2
    assert tool_names.count("execute_readonly_sql") == 1
    # The invalid mixed-period SQL is corrected before the result presentation.
    assert llm.calls == [
        "SQLGenerationResult",
        "SQLGenerationResult",
        "ResultPresentation",
    ]


def test_cross_channel_product_ranking_is_generated_from_live_metadata():
    metadata = _metadata()
    llm = FakeLLMClient(
        sql="""WITH channel_sales AS (
          SELECT ProductKey, SalesAmount FROM FactInternetSales
          UNION ALL
          SELECT ProductKey, SalesAmount FROM FactResellerSales
        )
        SELECT p.ProductName, SUM(s.SalesAmount) AS TotalSales
        FROM channel_sales s JOIN DimProduct p ON p.ProductKey = s.ProductKey
        GROUP BY p.ProductKey, p.ProductName
        ORDER BY TotalSales DESC LIMIT 10"""
    )
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
    # No packaged-schema shortcut: the same metadata-driven path handles every DB.
    assert llm.sql_call_count == 1


@pytest.mark.parametrize(
    "question",
    [
        "what are the most sold products by month in 2014",
        "Top selling product in each month in 2014",
    ],
)
def test_monthly_product_winner_executes_without_false_global_ranking_retry(question):
    metadata = _metadata()
    sql = """WITH InternetSales AS (
      SELECT ProductKey, strftime('%Y-%m', OrderDate) AS SalesMonth,
             SUM(OrderQuantity) AS TotalQty
      FROM FactInternetSales
      WHERE OrderDate >= '2014-01-01' AND OrderDate < '2015-01-01'
      GROUP BY ProductKey, strftime('%Y-%m', OrderDate)
    ), ResellerSales AS (
      SELECT ProductKey, strftime('%Y-%m', OrderDate) AS SalesMonth,
             SUM(OrderQuantity) AS TotalQty
      FROM FactResellerSales
      WHERE OrderDate >= '2014-01-01' AND OrderDate < '2015-01-01'
      GROUP BY ProductKey, strftime('%Y-%m', OrderDate)
    ), Combined AS (
      SELECT ProductKey, SalesMonth, SUM(TotalQty) AS TotalQty
      FROM (
        SELECT ProductKey, SalesMonth, TotalQty FROM InternetSales
        UNION ALL
        SELECT ProductKey, SalesMonth, TotalQty FROM ResellerSales
      )
      GROUP BY ProductKey, SalesMonth
    ), Ranked AS (
      SELECT c.SalesMonth, c.ProductKey, dp.ProductName, c.TotalQty,
             ROW_NUMBER() OVER (
               PARTITION BY c.SalesMonth ORDER BY c.TotalQty DESC
             ) AS rn
      FROM Combined AS c
      JOIN DimProduct AS dp ON dp.ProductKey = c.ProductKey
    )
    SELECT SalesMonth, ProductKey, ProductName, TotalQty
    FROM Ranked WHERE rn = 1 ORDER BY SalesMonth"""
    llm = FakeLLMClient(
        sql=sql,
        relevant_tables=["DimProduct", "FactInternetSales", "FactResellerSales"],
    )
    workflow = build_workflow(llm)

    final_state = workflow.invoke(_initial_state(question, metadata))

    response = final_state["final_response"]
    assert response["status"] == "ok"
    assert response["retry_count"] == 0
    assert llm.sql_call_count == 1
    assert response["time_context"]["target_year"] == 2014
    assert response["dataframe"].to_dict("records") == [
        {
            "SalesMonth": "2014-01",
            "ProductKey": 528,
            "ProductName": "Mountain Tire Tube",
            "TotalQty": 166,
        }
    ]


def test_missing_global_ranking_clauses_are_locally_repaired_without_llm_retry():
    metadata = _metadata()
    llm = FakeLLMClient(
        sql=(
            "SELECT p.ProductName, SUM(f.OrderQuantity) AS TotalUnits "
            "FROM FactInternetSales f "
            "JOIN DimProduct p ON p.ProductKey = f.ProductKey "
            "GROUP BY p.ProductKey, p.ProductName"
        ),
        relevant_tables=["DimProduct", "FactInternetSales"],
    )
    workflow = build_workflow(llm)

    final_state = workflow.invoke(
        _initial_state("Show the top 3 products by total OrderQuantity", metadata)
    )

    response = final_state["final_response"]
    assert response["status"] == "ok"
    assert response["retry_count"] == 0
    assert llm.sql_call_count == 1
    assert "ORDER BY TotalUnits DESC" in response["execution_sql"]
    assert "LIMIT 3" in response["execution_sql"]
    assert len(response["dataframe"]) == 3
    assert "Applied descending metric ordering" in response["sql_explanation"]


def test_grouped_extrema_executes_without_being_forced_into_global_ranking():
    metadata = _metadata()
    llm = FakeLLMClient(
        sql=(
            "SELECT ProductLine, MAX(ListPrice) AS HighestListPrice "
            "FROM DimProduct GROUP BY ProductLine ORDER BY ProductLine"
        ),
        relevant_tables=["DimProduct"],
    )
    workflow = build_workflow(llm)

    final_state = workflow.invoke(
        _initial_state("What is the highest ListPrice by ProductLine?", metadata)
    )

    response = final_state["final_response"]
    assert response["status"] == "ok"
    assert response["retry_count"] == 0
    assert llm.sql_call_count == 1
    assert set(response["dataframe"].columns) == {
        "ProductLine",
        "HighestListPrice",
    }


def test_raw_cross_fact_fanout_is_corrected_before_execution():
    metadata = _metadata()
    unsafe_fanout = """SELECT p.ProductName,
      SUM(i.OrderQuantity) + SUM(r.OrderQuantity) AS TotalUnits
      FROM FactInternetSales i
      JOIN DimProduct p ON p.ProductKey = i.ProductKey
      JOIN FactResellerSales r ON r.ProductKey = p.ProductKey
      GROUP BY p.ProductKey, p.ProductName"""
    safe_sql = """WITH all_sales AS (
      SELECT ProductKey, OrderQuantity FROM FactInternetSales
      UNION ALL
      SELECT ProductKey, OrderQuantity FROM FactResellerSales
    )
    SELECT p.ProductName, SUM(s.OrderQuantity) AS TotalUnits
    FROM all_sales s JOIN DimProduct p ON p.ProductKey = s.ProductKey
    GROUP BY p.ProductKey, p.ProductName ORDER BY p.ProductName"""
    llm = FakeLLMClient(
        sql=safe_sql,
        bad_sql=unsafe_fanout,
        fail_first_n_sql=1,
        relevant_tables=["DimProduct", "FactInternetSales", "FactResellerSales"],
    )
    workflow = build_workflow(llm)

    final_state = workflow.invoke(
        _initial_state("Total OrderQuantity across both channels by product", metadata)
    )

    response = final_state["final_response"]
    assert response["status"] == "ok"
    assert response["retry_count"] == 1
    assert llm.sql_call_count == 2
    assert len(response["dataframe"]) > 0


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


def test_provider_rate_limit_stops_after_intent_instead_of_calling_sql_again():
    class RateLimitedLLM(FakeLLMClient):
        def complete_json(self, **_kwargs):
            self.calls.append("rate_limited")
            raise LLMRateLimitError("OpenRouter rate limit: HTTP 429")

    metadata = _metadata()
    llm = RateLimitedLLM(sql=GOOD_SQL)
    workflow = build_workflow(llm)

    with pytest.raises(LLMRateLimitError):
        workflow.invoke(_initial_state("monthly internet revenue", metadata))

    assert llm.calls == ["rate_limited"]
    assert llm.sql_call_count == 0
