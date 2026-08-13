"""LangGraph assembly for the analytics workflow.

Pipeline: question -> retrieve relevant metadata -> generate SQL -> validate SQL
-> execute SQL -> combined result insight + visualization plan -> generate chart
-> respond.

Validation/execution failures route to a bounded correction loop
(``handle_error`` -> back to ``generate_sql``) instead of straight to
failure, up to ``max_retries`` attempts.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.state import AnalystState
from app.llm.client import LLMClient


def _configure_langchain_runtime() -> None:
    """Initialize legacy globals through LangChain Core's public API.

    LangGraph 0.2 supports ``langchain-core`` 0.3 and does not require the
    top-level ``langchain`` package. If a newer top-level LangChain happens to
    be installed in the environment, Core 0.3 detects it and reads the legacy
    ``debug``/``verbose``/``llm_cache`` module attributes that LangChain 1.x no
    longer defines. Calling the public setters creates those compatibility
    attributes and keeps callback initialization provider-neutral.
    """
    from langchain_core.globals import set_debug, set_llm_cache, set_verbose

    set_debug(False)
    set_verbose(False)
    set_llm_cache(None)


def build_workflow(llm_client: LLMClient):
    _configure_langchain_runtime()
    graph = StateGraph(AnalystState)

    graph.add_node(
        "retrieve_metadata", nodes.make_retrieve_relevant_metadata_node(llm_client)
    )
    graph.add_node("generate_sql", nodes.make_generate_sql_node(llm_client))
    graph.add_node("validate_sql", nodes.validate_sql_node)
    graph.add_node("execute_sql", nodes.execute_sql_node)
    graph.add_node("handle_error", nodes.handle_error_node)
    graph.add_node("give_up", nodes.give_up_node)
    graph.add_node("analyze_results", nodes.make_analyze_results_node(llm_client))
    graph.add_node("generate_chart", nodes.generate_chart_node)
    graph.add_node("respond", nodes.respond_node)

    # Schema RAG already searches the user's natural-language question, so a
    # separate LLM intent-classification call only duplicated work and consumed
    # provider quota. Start with deterministic retrieval instead.
    graph.add_edge(START, "retrieve_metadata")
    graph.add_edge("retrieve_metadata", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")

    graph.add_conditional_edges(
        "validate_sql",
        nodes.route_after_validation,
        {"execute_sql": "execute_sql", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "execute_sql",
        nodes.route_after_execution,
        {"analyze_results": "analyze_results", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "handle_error",
        nodes.route_after_error,
        {"generate_sql": "generate_sql", "give_up": "give_up"},
    )

    graph.add_edge("give_up", END)
    # analyze_results returns both the narrative and chart plan in one LLM call.
    graph.add_edge("analyze_results", "generate_chart")
    graph.add_edge("generate_chart", "respond")
    graph.add_edge("respond", END)

    return graph.compile()
