"""Node implementations for the analytics LangGraph workflow.

Every node is a plain function ``(state) -> partial_state_update``. LLM calls
happen only in ``generate_sql`` and the combined result-presentation node;
every other node
(metadata retrieval, validation, execution, chart rendering) is pure,
deterministic backend logic. Node factories close over an
:class:`~app.llm.client.LLMClient` so the graph is easy to build once per
process and easy to test with a fake client.
"""

from __future__ import annotations

from typing import Any

import json

from app.analysis.insights import (
    MAX_PREVIEW_ROWS,
    deterministic_direct_answer,
    enforce_numeric_grounding,
    generate_result_presentation,
    summarize_dataframe,
)
from app.config import get_settings
from app.graph.state import AnalystState
from app.llm.client import LLMClient, LLMError
from app.llm.schemas import IntentResult, VisualizationPlan
from app.logging_config import get_logger
from app.mcp_client.database import call_database_mcp_tool
from app.metadata import retrieval
from app.observability import trace_span
from app.sql.formatter import format_sql_for_display
from app.sql.generator import generate_sql
from app.sql.time_context import (
    format_time_context_for_prompt,
    question_requires_time_context,
)
from app.viz.planner import fallback_plan, sanitize_plan
from app.viz.renderer import render_chart

logger = get_logger(__name__)

def _format_history(state: AnalystState, limit: int = 3) -> str:
    history = (state.get("conversation_history") or [])[-limit:]
    if not history:
        return ""
    lines = ["RECENT CONVERSATION (most recent last):"]
    for i, turn in enumerate(history, start=1):
        lines.append(f"  {i}. Q: {turn.get('question', '')}")
        if turn.get("sql"):
            lines.append(f"     SQL: {turn['sql']}")
        if turn.get("result_columns"):
            columns = ", ".join(str(column) for column in turn["result_columns"])
            lines.append(f"     Result columns: {columns}")
        if turn.get("row_count") is not None:
            lines.append(f"     Result rows: {turn['row_count']}")
    return "\n".join(lines)


def _prior_error(state: AnalystState) -> str | None:
    validation_errors = state.get("validation_errors") or []
    if validation_errors:
        return "; ".join(validation_errors)
    return state.get("execution_error")


def _retrieve_relevant_metadata(
    state: AnalystState, llm_client: LLMClient | None = None
) -> dict[str, Any]:
    settings = get_settings()
    intent = state.get("intent") or {}
    hinted_tables = intent.get("relevant_tables") or []
    invocation = call_database_mcp_tool(
        "search_schema",
        {
            "question": state["question"],
            "hinted_tables": hinted_tables,
            "top_k": settings.vector.top_k,
        },
        stage="schema_retrieval",
    )
    relevant = invocation.value["metadata"]
    mode = invocation.value["mode"]
    records = [*(state.get("tool_records") or []), invocation.record]
    if invocation.value.get("ambiguous") and llm_client is not None:
        catalog = [
            {
                "name": name,
                "kind": table.get("kind", "unknown"),
                "object_type": table.get("object_type", "table"),
                "description": table.get("description", ""),
            }
            for name, table in state.get("metadata", {}).get("tables", {}).items()
        ]
        intent = llm_client.complete_json(
            system_prompt=(
                "Choose the smallest relevant table set for a data question from the "
                "exact live catalog. Never invent table names. This is retrieval only; "
                "do not answer the question or write SQL."
            ),
            user_prompt=(
                f"LIVE TABLE CATALOG:\n{json.dumps(catalog, default=str)}\n\n"
                f"QUESTION:\n{state['question']}"
            ),
            schema=IntentResult,
        )
        canonical = {
            name.casefold(): name
            for name in state.get("metadata", {}).get("tables", {})
        }
        hints = [
            canonical[name.casefold()]
            for name in intent.relevant_tables
            if name.casefold() in canonical
        ]
        if hints:
            refined = call_database_mcp_tool(
                "search_schema",
                {
                    "question": state["question"],
                    "hinted_tables": hints,
                    "top_k": settings.vector.top_k,
                },
                stage="schema_retrieval_disambiguation",
            )
            relevant = refined.value["metadata"]
            mode = f"{refined.value['mode']}+catalog"
            records.append(refined.record)
    time_context: dict[str, Any] = {"applied": False}
    if question_requires_time_context(state["question"]):
        time_invocation = call_database_mcp_tool(
            "resolve_relative_time",
            {
                "question": state["question"],
                "table_names": list(relevant.get("tables", {})),
            },
            stage="time_context_resolution",
        )
        time_context = time_invocation.value
        records.append(time_invocation.record)
    metadata_text = retrieval.format_metadata_for_prompt(relevant)
    time_text = format_time_context_for_prompt(time_context)
    if time_text:
        metadata_text = f"{metadata_text}\n\n{time_text}"
    return {
        "relevant_metadata": relevant,
        "metadata_text": metadata_text,
        "retrieval_mode": mode,
        "time_context": time_context,
        "tool_records": records,
    }


def retrieve_relevant_metadata(state: AnalystState) -> dict[str, Any]:
    """Deterministic retrieval entry point retained for direct/offline callers."""
    return _retrieve_relevant_metadata(state)


def make_retrieve_relevant_metadata_node(llm_client: LLMClient):
    def retrieve(state: AnalystState) -> dict[str, Any]:
        return _retrieve_relevant_metadata(state, llm_client)

    return retrieve


def make_generate_sql_node(llm_client: LLMClient):
    def generate_sql_node(state: AnalystState) -> dict[str, Any]:
        with trace_span(
            "generate_sql",
            category="agent_stage",
            metadata={
                "provider": llm_client.provider_name,
                "model": llm_client.model_name,
                "correction": bool(_prior_error(state)),
                "schema_chars": len(state.get("metadata_text") or ""),
                "schema_tables": len(
                    (state.get("relevant_metadata") or {}).get("tables", {})
                ),
            },
        ):
            result = generate_sql(
                llm_client,
                question=state["question"],
                metadata_text=state["metadata_text"] or "",
                prior_sql=state.get("sql"),
                prior_error=_prior_error(state),
                conversation_history_text=_format_history(state),
            )
        return {"sql": result.sql, "sql_explanation": result.explanation}

    return generate_sql_node


def validate_sql_node(state: AnalystState) -> dict[str, Any]:
    invocation = call_database_mcp_tool(
        "validate_readonly_sql",
        {
            "sql": state.get("sql") or "",
            "allowed_tables": list(
                (state.get("relevant_metadata") or {}).get("tables", {})
            ),
            "time_context": state.get("time_context") or {},
            "question": state["question"],
        },
        stage="sql_validation",
    )
    result = invocation.value
    records = [*(state.get("tool_records") or []), invocation.record]
    if not result.is_valid:
        logger.warning("SQL validation failed: %s", result.errors)
        return {
            "validation_errors": result.errors,
            "sanitized_sql": None,
            "download_sql": None,
            "tool_records": records,
        }
    return {
        "validation_errors": [],
        "sanitized_sql": result.sanitized_sql,
        "download_sql": result.download_sql,
        "sql_explanation": " ".join(
            filter(None, [state.get("sql_explanation"), *result.repairs])
        ),
        "tool_records": records,
    }


def route_after_validation(state: AnalystState) -> str:
    return "execute_sql" if not state.get("validation_errors") else "handle_error"


def execute_sql_node(state: AnalystState) -> dict[str, Any]:
    invocation = call_database_mcp_tool(
        "execute_readonly_sql",
        {"sql": state["sanitized_sql"]},
        stage="query_execution",
    )
    result = invocation.value
    records = [*(state.get("tool_records") or []), invocation.record]
    if not result.success:
        logger.warning("SQL execution failed: %s", result.error)
        return {
            "execution_error": result.error,
            "dataframe": None,
            "tool_records": records,
        }
    duplicate_columns = [
        str(column)
        for column in result.dataframe.columns
        if list(result.dataframe.columns).count(column) > 1
    ]
    if duplicate_columns:
        names = ", ".join(dict.fromkeys(duplicate_columns))
        return {
            "execution_error": (
                "The query returned duplicate output column names "
                f"({names}). Give every selected expression a unique, meaningful alias."
            ),
            "dataframe": None,
            "tool_records": records,
        }
    return {
        "execution_error": None,
        "dataframe": result.dataframe,
        "row_count": result.row_count,
        "truncated": result.truncated,
        "tool_records": records,
    }


def route_after_execution(state: AnalystState) -> str:
    return "analyze_results" if not state.get("execution_error") else "handle_error"


def handle_error_node(state: AnalystState) -> dict[str, Any]:
    retry_count = state.get("retry_count", 0) + 1
    logger.info(
        "Routing to correction (attempt %d/%d): %s",
        retry_count,
        state.get("max_retries", get_settings().limits.max_retries),
        _prior_error(state),
    )
    return {"retry_count": retry_count}


def route_after_error(state: AnalystState) -> str:
    max_retries = state.get("max_retries", get_settings().limits.max_retries)
    if state.get("retry_count", 0) <= max_retries:
        return "generate_sql"
    return "give_up"


def give_up_node(state: AnalystState) -> dict[str, Any]:
    error_message = _prior_error(state) or "Unknown error."
    error_stage = "validation" if state.get("validation_errors") else "execution"
    return {
        "status": "error",
        "final_response": {
            "status": "error",
            "question": state["question"],
            "sql": format_sql_for_display(state.get("sql")),
            "error": error_message,
            "error_stage": error_stage,
            "retry_count": state.get("retry_count", 0),
            "tool_records": state.get("tool_records", []),
        },
    }


def make_analyze_results_node(llm_client: LLMClient):
    def analyze_results(state: AnalystState) -> dict[str, Any]:
        df = state["dataframe"]
        result_summary = summarize_dataframe(df)
        summary_chars = len(json.dumps(result_summary, default=str))
        try:
            with trace_span(
                "analyze_and_plan_results",
                category="agent_stage",
                metadata={
                    "rows": len(df),
                    "raw_preview_rows_sent": min(len(df), MAX_PREVIEW_ROWS),
                    "result_summary_chars": summary_chars,
                    "model": llm_client.model_name,
                    "combined_outputs": ["insight", "visualization"],
                },
            ):
                presentation = generate_result_presentation(
                    llm_client,
                    question=state["question"],
                    sql=state["sanitized_sql"],
                    df=df,
                    result_summary=result_summary,
                )
            plan = sanitize_plan(presentation.visualization, df)
            grounded = deterministic_direct_answer(
                state["question"], df, sql=state["sanitized_sql"]
            )
            insight = grounded or enforce_numeric_grounding(
                presentation.insight,
                question=state["question"],
                df=df,
            )
            return {
                "insight": insight.model_dump(),
                "viz_plan": plan.model_dump(),
            }
        except LLMError:
            logger.exception(
                "Combined insight/chart generation failed; using deterministic fallbacks"
            )
            plan = fallback_plan(df, title=state["question"][:60])
            return {
                "insight": {
                    "summary": (
                        f"Query returned {len(df)} row(s). "
                        "(AI narrative unavailable: LLM call failed.)"
                    ),
                    "key_findings": [],
                },
                "viz_plan": plan.model_dump(),
            }

    return analyze_results


def generate_chart_node(state: AnalystState) -> dict[str, Any]:
    df = state["dataframe"]
    plan = sanitize_plan(VisualizationPlan(**state["viz_plan"]), df)
    fig = render_chart(plan, df)
    return {"viz_plan": plan.model_dump(), "chart": fig}


def respond_node(state: AnalystState) -> dict[str, Any]:
    return {
        "status": "ok",
        "final_response": {
            "status": "ok",
            "question": state["question"],
            "sql": format_sql_for_display(state.get("sanitized_sql")),
            "execution_sql": state.get("sanitized_sql"),
            "download_sql": state.get("download_sql"),
            "sql_explanation": state.get("sql_explanation"),
            "dataframe": state.get("dataframe"),
            "row_count": state.get("row_count", 0),
            "truncated": state.get("truncated", False),
            "insight": state.get("insight"),
            "viz_plan": state.get("viz_plan"),
            "chart": state.get("chart"),
            "retry_count": state.get("retry_count", 0),
            "retrieval_mode": state.get("retrieval_mode"),
            "time_context": state.get("time_context"),
            "tool_records": state.get("tool_records", []),
        },
    }
