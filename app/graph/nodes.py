"""Node implementations for the analytics LangGraph workflow.

Every node is a plain function ``(state) -> partial_state_update``. LLM
calls happen only in ``understand_intent``, ``generate_sql``,
``analyze_results`` and ``plan_visualization``; every other node
(metadata retrieval, validation, execution, chart rendering) is pure,
deterministic backend logic. Node factories close over an
:class:`~app.llm.client.LLMClient` so the graph is easy to build once per
process and easy to test with a fake client.
"""

from __future__ import annotations

import json
from typing import Any

from app.analysis.insights import generate_insight
from app.config import get_settings
from app.db.connection import get_active_database_identity
from app.db.executor import execute_sql
from app.graph.state import AnalystState
from app.llm.client import LLMClient, LLMError
from app.llm.schemas import IntentResult, VisualizationPlan
from app.logging_config import get_logger
from app.metadata import retrieval, store
from app.sql.formatter import format_sql_for_display
from app.sql.generator import generate_sql
from app.sql.validator import validate_sql
from app.viz.planner import fallback_plan, plan_visualization, sanitize_plan
from app.viz.renderer import render_chart

logger = get_logger(__name__)

_INTENT_SYSTEM_PROMPT = """You are the intent-understanding stage of an analytics
platform. Given a user's natural-language question and a catalog of the tables
available (name, kind, description, row count -- no columns yet), classify the
question and name which tables are likely relevant. This is a coarse first pass;
you do not need column-level detail yet.

You may also be given a short list of recent questions from the same session
(most recent last). Use it only to resolve follow-ups that refer back to it
(e.g. "now break that down by year", "same but for resellers") -- if the new
question stands on its own, ignore the history.

Use the exact response keys intent_summary, analysis_type, relevant_tables,
metrics, filters_mentioned, and time_range_mentioned. Do not rename
analysis_type to classification or omit intent_summary.
"""


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


def make_understand_intent_node(llm_client: LLMClient):
    def understand_intent(state: AnalystState) -> dict[str, Any]:
        catalog = store.get_table_catalog(state["metadata"])
        history_block = _format_history(state)
        user_prompt = (
            f"QUESTION:\n{state['question']}\n\n"
            f"TABLE CATALOG (JSON):\n{json.dumps(catalog)}"
            + (f"\n\n{history_block}" if history_block else "")
        )
        try:
            intent: IntentResult = llm_client.complete_json(
                system_prompt=_INTENT_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=IntentResult,
            )
            return {"intent": intent.model_dump()}
        except LLMError:
            logger.exception("Intent understanding failed; proceeding without a hint")
            return {"intent": None}

    return understand_intent


def retrieve_relevant_metadata(state: AnalystState) -> dict[str, Any]:
    settings = get_settings()
    intent = state.get("intent") or {}
    hinted_tables = intent.get("relevant_tables") or []
    db_identity = get_active_database_identity() if settings.vector.enabled else None
    relevant, mode = retrieval.get_relevant_metadata_with_mode(
        state["metadata"],
        state["question"],
        hinted_tables=hinted_tables,
        db_identity=db_identity,
        top_k=settings.vector.top_k,
    )
    return {
        "relevant_metadata": relevant,
        "metadata_text": retrieval.format_metadata_for_prompt(relevant),
        "retrieval_mode": mode,
    }


def make_generate_sql_node(llm_client: LLMClient):
    def generate_sql_node(state: AnalystState) -> dict[str, Any]:
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
    settings = get_settings()
    allowed_tables = set((state.get("relevant_metadata") or {}).get("tables", {}).keys())
    result = validate_sql(
        state.get("sql") or "",
        allowed_tables,
        settings.limits.max_rows,
        download_max_rows=settings.limits.download_max_rows,
    )
    if not result.is_valid:
        logger.warning("SQL validation failed: %s", result.errors)
        return {
            "validation_errors": result.errors,
            "sanitized_sql": None,
            "download_sql": None,
        }
    return {
        "validation_errors": [],
        "sanitized_sql": result.sanitized_sql,
        "download_sql": result.download_sql,
    }


def route_after_validation(state: AnalystState) -> str:
    return "execute_sql" if not state.get("validation_errors") else "handle_error"


def execute_sql_node(state: AnalystState) -> dict[str, Any]:
    settings = get_settings()
    result = execute_sql(
        state["sanitized_sql"],
        max_rows=settings.limits.max_rows,
        timeout_seconds=settings.limits.statement_timeout_seconds,
    )
    if not result.success:
        logger.warning("SQL execution failed: %s", result.error)
        return {"execution_error": result.error, "dataframe": None}
    return {
        "execution_error": None,
        "dataframe": result.dataframe,
        "row_count": result.row_count,
        "truncated": result.truncated,
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
        },
    }


def make_analyze_results_node(llm_client: LLMClient):
    def analyze_results(state: AnalystState) -> dict[str, Any]:
        df = state["dataframe"]
        try:
            insight = generate_insight(
                llm_client, question=state["question"], sql=state["sanitized_sql"], df=df
            )
            return {"insight": insight.model_dump()}
        except LLMError:
            logger.exception("Insight generation failed")
            return {
                "insight": {
                    "summary": (
                        f"Query returned {len(df)} row(s). "
                        "(AI narrative unavailable: LLM call failed.)"
                    ),
                    "key_findings": [],
                }
            }

    return analyze_results


def make_plan_visualization_node(llm_client: LLMClient):
    def plan_viz(state: AnalystState) -> dict[str, Any]:
        df = state["dataframe"]
        try:
            plan = plan_visualization(llm_client, question=state["question"], df=df)
        except LLMError:
            logger.exception("Visualization planning failed; using deterministic fallback")
            plan = fallback_plan(df, title=state["question"][:60])
        return {"viz_plan": plan.model_dump()}

    return plan_viz


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
        },
    }
