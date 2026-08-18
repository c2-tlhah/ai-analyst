"""AI Analyst -- Streamlit presentation layer.

This file is intentionally "dumb": it only collects user input, calls
:func:`app.orchestrator.answer_question`, and renders the structured
:class:`~app.orchestrator.AnalysisResponse` it gets back. It never talks to
the LLM, the database, or LangGraph directly, and it never generates or
displays any chart-generation Python code -- everything analytical lives in
``app/``. Caching/session-memory decisions (metadata TTL cache, exact-answer
cache, token accounting) also live in the backend (``app/orchestrator.py``,
``app/llm/client.py``); this file only *displays* what those report.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

# Allow `streamlit run ui/streamlit_app.py` to find the `app` package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.config import get_settings
from app.llm.presentation import (
    operation_status,
    provider_label as provider_display_label,
    provider_runtime_note,
)
from app.mcp_client import FileAssistantResponse, answer_filesystem_question
from app.orchestrator import (
    AnalysisResponse,
    KnowledgeAnswerResponse,
    answer_knowledge_question,
    answer_question,
    clear_session_caches,
    check_llm_provider_health,
    connect_database,
    generate_ai_result_chart,
    generate_result_chart,
    get_active_database_info,
    get_error_guidance,
    get_example_questions,
    get_knowledge_base_documents,
    get_llm_catalog,
    get_provider_rate_limit_status,
    get_session_stats,
    get_table_catalog,
    inspect_chart_options,
    list_knowledge_base_versions,
    prepare_complete_download,
    rebuild_active_knowledge_base,
    refresh_metadata,
)
from app.observability import export_recent_traces, get_recent_trace_events, read_log_tail
from app.services import ServiceStatus, ensure_ollama_running

st.set_page_config(page_title="AI Analyst", page_icon="🌱", layout="wide")

_HISTORY_TURNS_FOR_MEMORY = 3
_HISTORY_TURNS_KEPT = 8

_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }

.ai-badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 500;
    margin-right: 0.4rem;
    margin-bottom: 0.3rem;
}
.ai-badge-cache { background: #E3F2E1; color: #1B5E20; }
.ai-badge-time { background: #EEF5EC; color: #33513A; }
.ai-badge-retry { background: #FFF3E0; color: #8A5300; }
.ai-badge-live { background: #E8F0FE; color: #1A4B8C; }

.ai-panel {
    border: 1px solid #DCE8DA;
    border-radius: 10px;
    padding: 0.75rem 0.9rem;
    background: #FAFDF9;
    margin-bottom: 0.6rem;
}
.ai-panel-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #4C6B52;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.ai-stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.86rem;
    padding: 0.1rem 0;
}
.ai-stat-row span:last-child { font-weight: 600; color: #1B2A1E; }

.ai-eco-note {
    font-size: 0.78rem;
    color: #4C6B52;
    line-height: 1.4;
    margin-top: 0.3rem;
}

.ai-question-card {
    border-left: 4px solid #2E7D32;
    padding: 0.2rem 0 0.2rem 0.9rem;
    margin-bottom: 0.6rem;
}
</style>
"""


def _render_actionable_issue(
    title: str,
    reason: str,
    suggestions: tuple[str, ...] | list[str] = (),
    *,
    level: str = "error",
) -> None:
    """Render one consistent reason + next-actions block throughout the UI."""
    message = f"**{title}**\n\n{reason}"
    renderer = {
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }.get(level, st.error)
    renderer(message)
    if suggestions:
        st.markdown("**Suggested next steps:**")
        for suggestion in suggestions:
            st.markdown(f"- {suggestion}")


def _render_tool_activity(records, *, label: str = "Tool activity") -> None:
    """Render a compact audit trail from backend tool records."""
    records = list(records or [])
    if not records:
        return

    def field(record, name, default=None):
        if isinstance(record, dict):
            return record.get(name, default)
        return getattr(record, name, default)

    completed = sum(field(record, "status") == "completed" for record in records)
    with st.expander(f"{label} · {completed}/{len(records)} completed", expanded=False):
        for index, record in enumerate(records, start=1):
            status = field(record, "status", "unknown")
            icon = "✅" if status == "completed" else "❌"
            name = field(record, "name", "unknown_tool")
            duration = int(field(record, "duration_ms", 0) or 0)
            transport = field(record, "transport", "internal")
            if transport == "mcp":
                st.caption("Transport: MCP · RAG/database gateway")
            st.markdown(f"**{index}. {icon} `{name}`** · {duration:,} ms")
            st.caption(field(record, "summary", "No summary was supplied."))
            arguments = field(record, "arguments", {}) or {}
            if arguments:
                st.json(arguments, expanded=False)
            error = field(record, "error")
            if error:
                st.error(error)


@st.cache_resource(show_spinner=False)
def _start_required_services() -> ServiceStatus:
    """Start local dependencies once for the lifetime of the Streamlit server."""
    try:
        return ensure_ollama_running()
    except Exception as exc:  # noqa: BLE001 - startup failures must leave the UI usable
        return ServiceStatus(
            name="Ollama",
            running=False,
            message=f"Ollama startup check failed: {exc}",
        )


def _init_state() -> None:
    # A widget's session_state value can't be reassigned after that widget
    # has rendered in the same run (Streamlit raises StreamlitAPIException),
    # so a successful connect stashes the resolved path here and it's
    # applied on the *next* run, before the db_source_input widget renders.
    if "_db_source_pending" in st.session_state:
        st.session_state.db_source_input = st.session_state.pop("_db_source_pending")
    if "_question_input_pending" in st.session_state:
        st.session_state.question_input = st.session_state.pop("_question_input_pending")
    if "history" not in st.session_state:
        st.session_state.history = []  # list[AnalysisResponse]
    if "question_input" not in st.session_state:
        st.session_state.question_input = ""
    if "_knowledge_input_pending" in st.session_state:
        st.session_state.knowledge_question_input = st.session_state.pop(
            "_knowledge_input_pending"
        )
    if "knowledge_question_input" not in st.session_state:
        st.session_state.knowledge_question_input = ""
    if "knowledge_history" not in st.session_state:
        st.session_state.knowledge_history = []  # list[KnowledgeAnswerResponse]
    if "file_question_input" not in st.session_state:
        st.session_state.file_question_input = ""
    if "file_history" not in st.session_state:
        st.session_state.file_history = []  # list[FileAssistantResponse]
    if "pending_file_question" not in st.session_state:
        st.session_state.pending_file_question = None
    if "pending_file_llm_provider" not in st.session_state:
        st.session_state.pending_file_llm_provider = None
    if "pending_file_llm_model" not in st.session_state:
        st.session_state.pending_file_llm_model = None
    if "pending_file_mutations" not in st.session_state:
        st.session_state.pending_file_mutations = False
    # Two-phase ask flow: a click only *requests* a question (busy=True,
    # pending_question set) and reruns immediately so the UI redraws with
    # buttons disabled before any slow work starts; the actual LLM pipeline
    # only runs on the following script pass. This closes the window where
    # a user could double-click "Ask" (or an example) mid-flight and fire
    # the ~15-40s pipeline twice for the same question.
    if "busy" not in st.session_state:
        st.session_state.busy = False
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "pending_llm_provider" not in st.session_state:
        st.session_state.pending_llm_provider = None
    if "pending_llm_model" not in st.session_state:
        st.session_state.pending_llm_model = None
    if "pending_conversation_history" not in st.session_state:
        st.session_state.pending_conversation_history = None
    if "pending_knowledge_question" not in st.session_state:
        st.session_state.pending_knowledge_question = None
    if "pending_knowledge_llm_provider" not in st.session_state:
        st.session_state.pending_knowledge_llm_provider = None
    if "pending_knowledge_llm_model" not in st.session_state:
        st.session_state.pending_knowledge_llm_model = None
    if "generated_charts" not in st.session_state:
        st.session_state.generated_charts = {}
    if "prepared_downloads" not in st.session_state:
        st.session_state.prepared_downloads = {}
    if "visible_recommended_charts" not in st.session_state:
        st.session_state.visible_recommended_charts = set()
    if "db_source_input" not in st.session_state:
        st.session_state.db_source_input = get_active_database_info()["path"]
    if "db_connect_result" not in st.session_state:
        st.session_state.db_connect_result = None


def _response_context(response: AnalysisResponse) -> dict:
    dataframe = response.dataframe
    return {
        "question": response.question,
        "sql": response.sql,
        "row_count": response.row_count,
        "result_columns": list(dataframe.columns) if dataframe is not None else [],
    }


def _response_key(response: AnalysisResponse) -> str:
    identity = "\n".join(
        [
            response.llm_provider or "",
            response.llm_model or "",
            response.question,
            response.sql or "",
        ]
    )
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
    # The same cached answer can appear more than once in history; object identity
    # keeps Streamlit widget keys unique while remaining stable across reruns.
    return f"{digest}-{id(response)}"


def _conversation_memory(context_response: AnalysisResponse | None = None) -> list[dict]:
    """Last few (question, sql) turns, oldest first -- fed back to the backend
    so short follow-up questions don't need to repeat context."""
    turns = []
    for response in reversed(st.session_state.history[:_HISTORY_TURNS_FOR_MEMORY]):
        if response.status == "ok":
            turns.append(_response_context(response))

    if context_response is not None and context_response.status == "ok":
        context = _response_context(context_response)
        turns = [turn for turn in turns if turn["question"] != context["question"]]
        turns.append(context)
    return turns[-_HISTORY_TURNS_FOR_MEMORY:]


def _request_question(
    question: str,
    llm_provider: str,
    llm_model: str | None,
    context_response: AnalysisResponse | None = None,
) -> None:
    """Phase 1: record intent to ask, disable inputs, redraw, then run."""
    if st.session_state.busy:
        return
    if not question.strip():
        guidance = get_error_guidance("Please enter a question.", stage="input")
        _render_actionable_issue(
            guidance.title, guidance.reason, guidance.suggestions, level="warning"
        )
        return
    if not llm_model:
        guidance = get_error_guidance(
            "No ready LLM model is selected.", stage="configuration"
        )
        _render_actionable_issue(guidance.title, guidance.reason, guidance.suggestions)
        return
    # Deferred: the question_input widget has already rendered this run, and
    # newer Streamlit raises if a widget-bound session_state key is set
    # after that (see _init_state's _question_input_pending handling).
    st.session_state._question_input_pending = question
    st.session_state.pending_question = question
    st.session_state.pending_llm_provider = llm_provider
    st.session_state.pending_llm_model = llm_model
    st.session_state.pending_conversation_history = _conversation_memory(context_response)
    st.session_state.busy = True
    st.rerun()


def _run_pending_question() -> None:
    """Phase 2: do the actual (slow) work, then release the busy guard."""
    question = st.session_state.pending_question
    try:
        with st.spinner(operation_status(
            st.session_state.pending_llm_provider,
            st.session_state.pending_llm_model,
            "data",
        )):
            response = answer_question(
                question,
                conversation_history=st.session_state.pending_conversation_history or [],
                llm_provider=st.session_state.pending_llm_provider,
                llm_model=st.session_state.pending_llm_model,
            )
    except Exception as exc:  # noqa: BLE001 - always release the UI busy state
        guidance = get_error_guidance(exc, stage="workflow")
        response = AnalysisResponse(
            status="error",
            question=question or "",
            error=guidance.reason,
            error_title=guidance.title,
            error_stage=guidance.stage,
            error_suggestions=guidance.suggestions,
            llm_provider=st.session_state.pending_llm_provider,
            llm_model=st.session_state.pending_llm_model,
        )
    st.session_state.history.insert(0, response)
    st.session_state.history = st.session_state.history[:_HISTORY_TURNS_KEPT]
    st.session_state.pending_question = None
    st.session_state.pending_llm_provider = None
    st.session_state.pending_llm_model = None
    st.session_state.pending_conversation_history = None
    st.session_state.busy = False
    st.rerun()


def _request_knowledge_question(
    question: str,
    llm_provider: str,
    llm_model: str | None,
) -> None:
    """Queue a document-grounded question and lock both question forms."""
    if st.session_state.busy:
        return
    if not question.strip():
        _render_actionable_issue(
            "Enter a knowledge-base question",
            "The question is empty.",
            (
                "Ask what a table, column, relationship, metric, or business term means.",
                "Use Query data for calculated values, rankings, and trends.",
            ),
            level="warning",
        )
        return
    if not llm_model:
        guidance = get_error_guidance(
            "No ready LLM model is selected.", stage="configuration"
        )
        _render_actionable_issue(guidance.title, guidance.reason, guidance.suggestions)
        return
    st.session_state._knowledge_input_pending = question
    st.session_state.pending_knowledge_question = question
    st.session_state.pending_knowledge_llm_provider = llm_provider
    st.session_state.pending_knowledge_llm_model = llm_model
    st.session_state.busy = True
    st.rerun()


def _run_pending_knowledge_question() -> None:
    """Run one queued document-RAG question and always release the busy guard."""
    question = st.session_state.pending_knowledge_question
    try:
        with st.spinner(operation_status(
            st.session_state.pending_knowledge_llm_provider,
            st.session_state.pending_knowledge_llm_model,
            "knowledge",
        )):
            response = answer_knowledge_question(
                question or "",
                llm_provider=st.session_state.pending_knowledge_llm_provider,
                llm_model=st.session_state.pending_knowledge_llm_model,
            )
    except Exception as exc:  # noqa: BLE001 - always release the UI busy state
        guidance = get_error_guidance(exc, stage="workflow")
        response = KnowledgeAnswerResponse(
            status="error",
            question=question or "",
            error=guidance.reason,
            error_title=guidance.title,
            error_suggestions=guidance.suggestions,
            llm_provider=st.session_state.pending_knowledge_llm_provider,
            llm_model=st.session_state.pending_knowledge_llm_model,
        )
    st.session_state.knowledge_history.insert(0, response)
    st.session_state.knowledge_history = st.session_state.knowledge_history[
        :_HISTORY_TURNS_KEPT
    ]
    st.session_state.pending_knowledge_question = None
    st.session_state.pending_knowledge_llm_provider = None
    st.session_state.pending_knowledge_llm_model = None
    st.session_state.busy = False
    st.rerun()


def _request_file_question(
    question: str,
    llm_provider: str,
    llm_model: str | None,
    *,
    allow_mutations: bool,
) -> None:
    """Queue one filesystem-MCP request behind the global busy guard."""
    if st.session_state.busy:
        return
    if not question.strip():
        _render_actionable_issue(
            "Enter a file question or action",
            "The request is empty.",
            ("Ask to list, search, read, summarize, create, edit, or move a file.",),
            level="warning",
        )
        return
    if not llm_model:
        _render_actionable_issue(
            "Select a ready model",
            "A tool-capable LLM is required for filesystem MCP actions.",
        )
        return
    st.session_state.pending_file_question = question.strip()
    st.session_state.pending_file_llm_provider = llm_provider
    st.session_state.pending_file_llm_model = llm_model
    st.session_state.pending_file_mutations = allow_mutations
    st.session_state.busy = True
    st.rerun()


def _run_pending_file_question() -> None:
    """Run one MCP tool loop and always release the global busy guard."""
    question = st.session_state.pending_file_question or ""
    try:
        with st.spinner(operation_status(
            st.session_state.pending_file_llm_provider,
            st.session_state.pending_file_llm_model,
            "filesystem",
        )):
            response = answer_filesystem_question(
                question,
                llm_provider=st.session_state.pending_file_llm_provider,
                llm_model=st.session_state.pending_file_llm_model,
                allow_mutations=bool(st.session_state.pending_file_mutations),
            )
    except Exception as exc:  # noqa: BLE001 - keep the Streamlit session usable
        response = FileAssistantResponse(
            status="error",
            question=question,
            error=str(exc),
            llm_provider=st.session_state.pending_file_llm_provider,
            llm_model=st.session_state.pending_file_llm_model,
        )
    st.session_state.file_history.insert(0, response)
    st.session_state.file_history = st.session_state.file_history[:_HISTORY_TURNS_KEPT]
    st.session_state.pending_file_question = None
    st.session_state.pending_file_llm_provider = None
    st.session_state.pending_file_llm_model = None
    st.session_state.pending_file_mutations = False
    st.session_state.busy = False
    st.rerun()


def _render_sidebar(ollama_status: ServiceStatus) -> tuple[str, str | None, bool]:
    settings = get_settings()
    provider_before_render = st.session_state.get(
        "llm_provider", settings.default_llm_provider
    )
    llm_catalog = get_llm_catalog(
        discover_ollama=provider_before_render == "ollama",
        discover_openrouter=provider_before_render == "openrouter",
        discover_nvidia=provider_before_render == "nvidia_nim",
    )
    with st.sidebar:
        st.title("🌱 AI Analyst")
        st.caption("Schema-aware analytics with a cached, reusable context layer.")

        st.subheader("LLM configuration")
        provider_keys = list(llm_catalog)
        configured_default = settings.default_llm_provider
        if configured_default not in provider_keys:
            configured_default = "azure_foundry"
        if st.session_state.get("llm_provider") not in provider_keys:
            st.session_state.llm_provider = configured_default

        busy = st.session_state.busy
        selected_provider = st.selectbox(
            "Provider",
            provider_keys,
            format_func=lambda value: llm_catalog[value]["label"],
            key="llm_provider",
            disabled=busy,
        )
        provider_info = llm_catalog[selected_provider]
        models = provider_info["models"]

        if selected_provider == "ollama":
            if ollama_status.running:
                detail = "started by this app" if ollama_status.started_by_app else "already running"
                st.caption(f"Ollama service: ready ({detail})")
            else:
                guidance = get_error_guidance(ollama_status.message, stage="service")
                _render_actionable_issue(
                    guidance.title,
                    guidance.reason,
                    guidance.suggestions,
                    level="warning",
                )
                if st.button("Retry starting Ollama", width="stretch", disabled=busy):
                    _start_required_services.clear()
                    ensure_ollama_running(force_start=True)
                    st.rerun()
            if st.button("Refresh Ollama models", width="stretch", disabled=busy):
                get_llm_catalog(refresh_ollama=True)
                st.rerun()
        elif selected_provider == "openrouter":
            if st.button("Refresh OpenRouter models", width="stretch", disabled=busy):
                get_llm_catalog(
                    discover_ollama=False,
                    refresh_openrouter=True,
                )
                st.rerun()
            if provider_info.get("catalog_error"):
                st.warning(
                    "Using the built-in model list because live OpenRouter metadata "
                    f"could not be refreshed: {provider_info['catalog_error']}"
                )
        elif selected_provider == "nvidia_nim":
            request_budget = get_provider_rate_limit_status("nvidia_nim") or {}
            st.caption(
                "NVIDIA request budget: "
                f"{int(request_budget.get('remaining_requests', 60))}/"
                f"{int(request_budget.get('limit', 60))} remaining in the local "
                "rolling minute. Requests are queued and spaced automatically."
            )
            refresh_col, health_col = st.columns(2)
            if refresh_col.button(
                "Refresh models",
                width="stretch",
                disabled=busy or not bool(settings.nvidia_nim.api_key),
                key="refresh_nvidia_models",
            ):
                get_llm_catalog(
                    discover_ollama=False,
                    refresh_nvidia=True,
                )
                st.rerun()
            if health_col.button(
                "Test connection",
                width="stretch",
                disabled=busy or not bool(settings.nvidia_nim.api_key),
                key="test_nvidia_connection",
            ):
                st.session_state.nvidia_health = check_llm_provider_health(
                    "nvidia_nim",
                    st.session_state.get("llm_model::nvidia_nim")
                    or settings.nvidia_nim.model,
                )
            if provider_info.get("catalog_error"):
                guidance = get_error_guidance(
                    provider_info["catalog_error"], stage="provider"
                )
                _render_actionable_issue(
                    "Live NVIDIA catalog unavailable; configured models remain selectable",
                    guidance.reason,
                    guidance.suggestions,
                    level="warning",
                )
            health = st.session_state.get("nvidia_health")
            if health:
                if health.get("ok"):
                    model_state = (
                        "available" if health.get("model_available", True) else "not in catalog"
                    )
                    st.success(
                        f"NVIDIA connection ready ({health.get('host')}); selected model "
                        f"is {model_state}. DNS: {len(health.get('addresses', []))} address(es)."
                    )
                else:
                    _render_actionable_issue(
                        health.get("error_title") or "NVIDIA connection check failed",
                        health.get("error") or "Unknown provider health error.",
                        health.get("suggestions") or (),
                        level="warning",
                    )

        selected_model: str | None = None
        if models:
            model_key = f"llm_model::{selected_provider}"
            configured_model = (
                {
                    "ollama": settings.ollama.model,
                    "azure_foundry": settings.azure_ai.model_deployment,
                    "openrouter": settings.openrouter.model,
                    "nvidia_nim": settings.nvidia_nim.model,
                }.get(selected_provider, "")
            )
            if st.session_state.get(model_key) not in models:
                st.session_state[model_key] = (
                    configured_model if configured_model in models else models[0]
                )
            selected_model = st.selectbox(
                "Model",
                models,
                format_func=(
                    lambda model_id: provider_info.get("model_details", {})
                    .get(model_id, {})
                    .get("name", model_id)
                    if selected_provider in {"openrouter", "nvidia_nim"}
                    else model_id
                ),
                key=model_key,
                disabled=busy,
            )

        if selected_provider == "azure_foundry" and selected_model:
            is_kimi = "kimi-k2.6" in selected_model.casefold()
            with st.expander("Model parameters & capabilities", expanded=False):
                st.code(selected_model, language=None)
                st.markdown(
                    "**Runtime:** Azure AI Foundry hosted deployment  \n"
                    f"**Reasoning model:** {'yes' if is_kimi else 'deployment dependent'}  \n"
                    f"**Native filesystem tool path:** "
                    f"{'supported' if is_kimi else 'not enabled for this deployment'}  \n"
                    "**Workflow structured output:** strict prompt + Pydantic validation"
                )
                st.markdown("**Parameters sent by this app**")
                st.code(
                    "\n".join(
                        [
                            f"temperature = {settings.azure_ai.temperature}",
                            f"max_tokens = {settings.azure_ai.max_tokens}",
                            "stream = false (validated workflow output)",
                            f"timeout = {settings.azure_ai.request_timeout_seconds}s",
                            (
                                f"api_version = {settings.azure_ai.api_version}"
                                if settings.azure_ai.api_version
                                else "api_version = service default"
                            ),
                        ]
                    ),
                    language=None,
                )
        elif selected_provider == "ollama" and selected_model:
            with st.expander("Model parameters & capabilities", expanded=False):
                st.code(selected_model, language=None)
                st.markdown(
                    "**Runtime:** local Ollama service  \n"
                    "**Privacy:** prompts stay on the configured Ollama host  \n"
                    "**Structured output:** native Ollama JSON Schema format  \n"
                    "**Filesystem tool loop:** not implemented for Ollama in this app"
                )
                st.markdown("**Parameters sent by this app**")
                st.code(
                    "\n".join(
                        [
                            f"temperature = {settings.ollama.temperature}",
                            f"num_predict = {settings.ollama.max_tokens}",
                            "stream = false (validated workflow output)",
                            f"timeout = {settings.ollama.request_timeout_seconds}s",
                            f"auto_start = {settings.ollama.auto_start}",
                            f"stop_on_exit = {settings.ollama.stop_on_exit}",
                        ]
                    ),
                    language=None,
                )
                st.caption(
                    "Model size, quantization, context window, and reasoning behavior "
                    "come from the downloaded Ollama model and its Modelfile."
                )
        elif selected_provider == "openrouter" and selected_model:
            model_details = provider_info.get("model_details", {}).get(
                selected_model, {}
            )
            with st.expander("Model parameters & capabilities", expanded=False):
                st.code(selected_model, language=None)
                if model_details.get("description"):
                    description = model_details["description"]
                    st.caption(
                        description[:360] + ("…" if len(description) > 360 else "")
                    )
                context_length = int(model_details.get("context_length") or 0)
                context_text = (
                    f"{context_length:,} tokens" if context_length else "Unavailable"
                )
                input_text = ", ".join(model_details.get("input_modalities") or []) or "Unknown"
                output_text = ", ".join(model_details.get("output_modalities") or []) or "Unknown"
                st.markdown(
                    f"**Context window:** {context_text}  \n"
                    f"**Input:** {input_text}  \n"
                    f"**Output:** {output_text}  \n"
                    f"**Reasoning:** "
                    f"{'supported' if model_details.get('reasoning_supported') else 'not advertised'}  \n"
                    f"**Native structured output:** "
                    f"{'supported' if model_details.get('structured_output_supported') else 'not advertised'}"
                )
                prompt_price = model_details.get("prompt_price")
                completion_price = model_details.get("completion_price")
                if prompt_price == "0" and completion_price == "0":
                    st.markdown("**Current catalog pricing:** free")
                supported = model_details.get("supported_parameters") or []
                st.markdown("**Supported OpenRouter parameters**")
                st.code(", ".join(supported) if supported else "Metadata unavailable", language=None)
                st.markdown("**Parameters sent by this app**")
                st.code(
                    "\n".join(
                        [
                            f"reasoning.enabled = {settings.openrouter.reasoning_enabled}",
                            f"temperature = {settings.openrouter.temperature}",
                            f"max_tokens = {settings.openrouter.max_tokens}",
                            "stream = false",
                            f"timeout = {settings.openrouter.request_timeout_seconds}s",
                            f"transient retries = {settings.openrouter.max_retries}",
                            f"retry backoff = {settings.openrouter.retry_backoff_seconds}s",
                        ]
                    ),
                    language=None,
                )
                if not model_details.get("structured_output_supported"):
                    st.caption(
                        "For workflow JSON, the app uses strict schema prompting, "
                        "validation, response healing, and one bounded repair attempt."
                    )
        elif selected_provider == "nvidia_nim" and selected_model:
            model_details = provider_info.get("model_details", {}).get(
                selected_model, {}
            )
            reasoning_active = bool(
                settings.nvidia_nim.reasoning_enabled
                and model_details.get("reasoning_controls_supported", True)
            )
            fixed_profile = bool(model_details.get("fixed_profile"))
            effective_max_tokens = (
                int(model_details.get("max_tokens"))
                if fixed_profile and model_details.get("max_tokens")
                else min(
                    settings.nvidia_nim.max_tokens,
                    int(
                        model_details.get("max_tokens")
                        or settings.nvidia_nim.max_tokens
                    ),
                )
            )
            effective_temperature = (
                model_details.get("temperature")
                if fixed_profile
                else settings.nvidia_nim.temperature
            )
            effective_top_p = (
                model_details.get("top_p")
                if fixed_profile
                else settings.nvidia_nim.top_p
            )
            with st.expander("Model parameters & capabilities", expanded=False):
                st.code(selected_model, language=None)
                st.markdown(
                    f"**Owner:** {model_details.get('owned_by') or 'NVIDIA'}  \n"
                    f"**Catalog status:** "
                    f"{'verified live' if model_details.get('verified') else 'configured fallback'}  \n"
                    f"**Reasoning model:** "
                    f"{'yes' if model_details.get('reasoning_supported') else 'not advertised'}  \n"
                    f"**Native tool calling:** "
                    f"{'supported' if model_details.get('tool_calling_supported') else 'not advertised'}  \n"
                    f"**Native JSON Schema:** "
                    f"{'verified' if model_details.get('structured_output_supported') else 'prompt validated'}"
                )
                st.markdown("**Parameters sent by this app**")
                st.code(
                    "\n".join(
                        [
                            f"temperature = {effective_temperature}",
                            f"top_p = {effective_top_p}",
                            f"max_tokens = {effective_max_tokens}",
                            (
                                f"seed = {model_details.get('seed')}"
                                if model_details.get("seed") is not None
                                else "seed = not sent"
                            ),
                            "stream = false (required for validated workflow output)",
                            "chat_template_kwargs.enable_thinking = "
                            f"{reasoning_active}",
                            (
                                f"reasoning_budget = {settings.nvidia_nim.reasoning_budget}"
                                if reasoning_active
                                else "reasoning_budget = not sent for this model"
                            ),
                            f"timeout = {settings.nvidia_nim.request_timeout_seconds}s",
                            f"transient retries = {settings.nvidia_nim.max_retries}",
                            f"shared request limit = {settings.nvidia_nim.requests_per_minute}/minute",
                            f"minimum request spacing = {settings.nvidia_nim.min_request_interval_seconds}s",
                            f"429 fallback cooldown = {settings.nvidia_nim.rate_limit_429_cooldown_seconds}s",
                            f"maximum local queue wait = {settings.nvidia_nim.rate_limit_max_wait_seconds}s",
                        ]
                    ),
                    language=None,
                )
                st.caption(
                    "A normal data question uses two model calls: SQL generation, "
                    "then one combined insight + chart-plan response."
                )

        llm_ready = bool(provider_info["available"] and selected_model)
        if llm_ready:
            st.success(
                f"Ready: {provider_display_label(selected_provider)} · {selected_model}"
            )
            st.caption(provider_runtime_note(selected_provider, selected_model))
        else:
            guidance = provider_info.get("error_guidance") or get_error_guidance(
                provider_info["error"] or "The selected LLM is unavailable.",
                stage="configuration",
            )
            _render_actionable_issue(
                guidance.title, guidance.reason, guidance.suggestions
            )

        st.divider()
        st.subheader("🔌 Database connection")
        if st.session_state.db_connect_result is not None:
            result = st.session_state.db_connect_result
            (st.success if result.success else st.error)(result.message)
            _render_tool_activity(result.tool_records, label="Database preparation tools")

        db_info = get_active_database_info()
        kb_status = db_info["vector_status"]
        if kb_status == "ready":
            st.caption(
                f"🔎 Knowledge base ready: {db_info['vector_document_count']} document(s), "
                f"version {db_info['vector_version']} of {db_info['vector_version_count']}."
            )
        elif kb_status in {"disabled", "documents_ready"}:
            _render_actionable_issue(
                "Documentation ready with lexical search",
                (
                    f"{db_info['vector_document_count']} generated document(s) are usable. "
                    "Vector embeddings are disabled, so searches use the document tool."
                ),
                (
                    "You can ask documentation questions now.",
                    "Enable VECTOR_RAG_ENABLED only when semantic similarity is useful for a large schema.",
                ),
                level="info",
            )
        elif kb_status == "error":
            _render_actionable_issue(
                "Semantic indexing failed; documentation remains available",
                db_info["vector_error"] or "The optional vector backend did not report a reason.",
                (
                    "Documentation questions automatically fall back to lexical file search.",
                    "Click Rebuild documentation to retry semantic indexing.",
                    "Run pip install -r requirements.txt in the active virtual environment.",
                    "Check logs/ai_analyst.log for the full backend traceback.",
                ),
                level="warning",
            )
        else:
            st.caption("🔎 Knowledge base has not been built for this database yet.")
        st.caption(f"Isolated metadata: {db_info['schema_metadata_path']}")

        versions = list_knowledge_base_versions()
        with st.expander("📚 Knowledge base explorer", expanded=False):
            if versions:
                version_options = [v["version"] for v in reversed(versions)]
                if st.session_state.get("kb_version_select") not in version_options:
                    st.session_state.kb_version_select = version_options[0]
                version_by_number = {v["version"]: v for v in versions}
                chosen_version = st.selectbox(
                    "Knowledge-base version",
                    version_options,
                    format_func=lambda v: (
                        f"v{v} · {version_by_number[v]['created_at'][:19]} "
                        f"({len(version_by_number[v]['tables'])} table(s))"
                        + (" · current" if v == db_info["vector_version"] else "")
                    ),
                    key="kb_version_select",
                    disabled=busy,
                )
                docs = get_knowledge_base_documents(chosen_version)
                document_filter = st.text_input(
                    "Filter documents",
                    key="kb_document_filter",
                    placeholder="Table, column, relationship, or business term",
                    disabled=busy,
                ).strip().casefold()
                visible_docs = {
                    name: text
                    for name, text in docs.items()
                    if not document_filter
                    or document_filter in name.casefold()
                    or document_filter in text.casefold()
                }
                combined_docs = "\n\n".join(
                    f"{'=' * 70}\n{name}\n{'=' * 70}\n{text}"
                    for name, text in sorted(docs.items())
                )
                st.download_button(
                    "Download all documents",
                    data=combined_docs.encode("utf-8"),
                    file_name=f"knowledge_base_v{chosen_version}.txt",
                    mime="text/plain",
                    width="stretch",
                    disabled=busy or not docs,
                    key=f"kb_download::{chosen_version}",
                )
                if not visible_docs:
                    st.info("No document matches that filter. Try a table or column name.")
                for table_name, text in visible_docs.items():
                    st.markdown(f"**{table_name}**")
                    st.text_area(
                        f"{table_name} knowledge document",
                        value=text,
                        height=220,
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"kb_doc::{chosen_version}::{table_name}",
                    )
                if db_info.get("vector_text_export_path"):
                    st.caption(f"Text export: {db_info['vector_text_export_path']}")
            else:
                st.info(
                    "No knowledge documents exist yet. Build the active database below; "
                    "an LLM is optional for schema indexing."
                )

        with st.form(key="db_connect_form", clear_on_submit=False):
            st.text_input(
                "SQLite file path or connection string",
                key="db_source_input",
                placeholder="e.g. C:\\data\\operations.db or sqlite:///data/local.db",
                disabled=busy,
            )
            connect_clicked = st.form_submit_button(
                "Connect & prepare database",
                disabled=busy,
            )
            st.caption(
                "Works with any populated SQLite schema. Audited tools derive "
                "tables, types, keys, relationships, semantics, and documentation "
                "from that database; no packaged business context is applied."
            )
        if connect_clicked:
            discovery_status = (
                operation_status(selected_provider, selected_model, "database")
                if llm_ready
                else "Running deterministic database discovery and documentation tools."
            )
            with st.spinner(discovery_status):
                connect_result = connect_database(
                    st.session_state.db_source_input,
                    llm_provider=selected_provider if llm_ready else None,
                    llm_model=selected_model if llm_ready else None,
                )
            st.session_state.db_connect_result = connect_result
            if connect_result.success and connect_result.db_path:
                st.session_state._db_source_pending = connect_result.db_path
                # Results, follow-ups, and charts belong to the previous
                # database and must never be shown against the new catalog.
                st.session_state.history = []
                st.session_state.knowledge_history = []
                st.session_state.generated_charts = {}
                st.session_state.visible_recommended_charts = set()
            st.rerun()

        if st.button(
            "Rebuild documentation",
            width="stretch",
            disabled=busy or not db_info["exists"],
        ):
            rebuild_status = (
                operation_status(selected_provider, selected_model, "database")
                if llm_ready
                else "Rerunning deterministic database preparation tools."
            )
            with st.spinner(rebuild_status):
                connect_result = rebuild_active_knowledge_base(
                    llm_provider=selected_provider if llm_ready else None,
                    llm_model=selected_model if llm_ready else None,
                )
            st.session_state.db_connect_result = connect_result
            st.rerun()

        st.divider()
        st.subheader("🌱 Session efficiency")
        stats = get_session_stats()
        meta_cache = stats["metadata_cache"]
        age = meta_cache.get("age_seconds")
        age_txt = f"{age:,.0f}s ago" if age is not None else "not yet checked"
        st.markdown(
            f"""<div class="ai-panel">
              <div class="ai-stat-row"><span>Schema last verified</span><span>{age_txt}</span></div>
              <div class="ai-stat-row"><span>Cached questions</span><span>{stats['cached_questions']}</span></div>
              <div class="ai-stat-row"><span>LLM calls made</span><span>{stats['llm_calls']}</span></div>
              <div class="ai-stat-row"><span>Total tokens used</span><span>{stats['total_tokens']:,}</span></div>
              <div class="ai-eco-note">Schema context is discovered once and reused for
              {int(meta_cache['ttl_seconds'] // 60)} minutes instead of being re-derived on every
              question; repeat questions are served from cache with zero extra LLM calls.</div>
            </div>""",
            unsafe_allow_html=True,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Refresh schema", width="stretch", disabled=busy):
                refresh_metadata(force=True)
                st.rerun()
        with col_b:
            if st.button("🧹 Clear cache", width="stretch", disabled=busy):
                clear_session_caches()
                st.rerun()

        st.divider()
        st.subheader("Available data")
        try:
            catalog = get_table_catalog()
            current_kb_docs = (
                get_knowledge_base_documents(db_info["vector_version"])
                if db_info["vector_version"]
                else {}
            )
            for table in catalog:
                with st.expander(f"{table['name']}  ·  {table['kind']}"):
                    st.caption(table["description"])
                    st.caption(f"~{table['row_count']:,} rows")
                    document = current_kb_docs.get(table["name"])
                    if document:
                        st.markdown("**Indexed knowledge document**")
                        st.text_area(
                            f"{table['name']} indexed knowledge",
                            value=document,
                            height=180,
                            disabled=True,
                            label_visibility="collapsed",
                            key=f"catalog_kb_doc::{db_info['vector_version']}::{table['name']}",
                        )
                        st.download_button(
                            "Download document",
                            data=document.encode("utf-8"),
                            file_name=f"{table['name']}.txt",
                            mime="text/plain",
                            width="stretch",
                            disabled=busy,
                            key=f"catalog_kb_download::{db_info['vector_version']}::{table['name']}",
                        )
                    elif kb_status != "ready":
                        st.caption("No indexed document yet; build the knowledge base above.")
        except Exception as exc:  # noqa: BLE001
            guidance = get_error_guidance(exc, stage="database")
            _render_actionable_issue(
                "The data catalog could not be loaded",
                guidance.reason,
                (
                    "Refresh the schema from the sidebar.",
                    "Verify that the selected SQLite file still exists and is readable.",
                    "Check logs/ai_analyst.log if the problem repeats.",
                ),
                level="warning",
            )

        st.divider()
        st.subheader("Try asking")
        for q in get_example_questions():
            if st.button(
                q,
                key=f"example::{q}",
                width="stretch",
                disabled=busy or not llm_ready,
            ):
                _request_question(q, selected_provider, selected_model)

        st.divider()
        if st.button("Clear history", width="stretch", disabled=busy):
            st.session_state.history = []
            st.session_state.knowledge_history = []
            st.session_state.generated_charts = {}
            st.session_state.visible_recommended_charts = set()
            st.rerun()

    return selected_provider, selected_model, llm_ready


def _render_badges(response: AnalysisResponse) -> None:
    badges = []
    if response.llm_model:
        provider_label = provider_display_label(response.llm_provider)
        badges.append(
            f'<span class="ai-badge ai-badge-time">{provider_label} · {response.llm_model}</span>'
        )
    if (response.retrieval_mode or "").startswith("vector"):
        badges.append('<span class="ai-badge ai-badge-cache">🔎 MCP + schema RAG · semantic</span>')
    elif (response.retrieval_mode or "").startswith("lexical"):
        badges.append('<span class="ai-badge ai-badge-time">🔤 MCP + schema RAG · lexical</span>')
    if response.cache_hit:
        badges.append('<span class="ai-badge ai-badge-cache">🔁 served from session cache</span>')
    elif response.status == "error":
        badges.append(
            f'<span class="ai-badge ai-badge-retry">failed after {response.elapsed_seconds:.1f}s</span>'
        )
    else:
        badges.append(
            f'<span class="ai-badge ai-badge-live">⚡ answered in {response.elapsed_seconds:.1f}s</span>'
        )
    if response.truncated:
        badges.append('<span class="ai-badge ai-badge-time">✂ result truncated to limit</span>')
    if response.retry_count:
        badges.append(
            f'<span class="ai-badge ai-badge-retry">↻ self-corrected {response.retry_count}x</span>'
        )
    if response.trace_id:
        badges.append(
            f'<span class="ai-badge ai-badge-time">trace {response.trace_id}</span>'
        )
    st.markdown("".join(badges), unsafe_allow_html=True)


@st.fragment(run_every=2.0)
def _render_live_diagnostics() -> None:
    """Auto-refresh a bounded, redacted view of current backend activity."""
    settings = get_settings()
    events = get_recent_trace_events(limit=500)
    trace_ids = list(
        dict.fromkeys(
            str(event.get("trace_id"))
            for event in reversed(events)
            if event.get("trace_id")
        )
    )
    selected_trace = st.selectbox(
        "Trace filter",
        ["All traces", *trace_ids],
        key="live_trace_filter",
        help="Choose one request trace to follow its nested agent, LLM, tool, and MCP stages.",
    )
    filtered = (
        events
        if selected_trace == "All traces"
        else [event for event in events if event.get("trace_id") == selected_trace]
    )
    rows = [
        {
            "time (UTC)": str(event.get("timestamp", ""))[11:23],
            "trace": str(event.get("trace_id", "")),
            "category": event.get("category"),
            "stage / tool": event.get("name"),
            "status": event.get("status"),
            "duration ms": event.get("duration_ms"),
            "message": event.get("message"),
        }
        for event in reversed(filtered[-250:])
    ]
    completed = sum(event.get("status") == "completed" for event in filtered)
    failed = sum(event.get("status") == "failed" for event in filtered)
    retrying = sum(event.get("status") == "retrying" for event in filtered)
    queued = sum(event.get("status") == "queued" for event in filtered)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Events", len(filtered))
    metric_cols[1].metric("Completed", completed)
    metric_cols[2].metric("Queued", queued)
    metric_cols[3].metric("Retries", retrying)
    metric_cols[4].metric("Failures", failed)

    if rows:
        st.dataframe(rows, width="stretch", height=430, hide_index=True)
        with st.expander("Latest structured event", expanded=False):
            st.json(filtered[-1])
    else:
        st.info("No trace events have been emitted in this process yet. Run a query or provider test.")

    st.caption(
        "This view refreshes every 2 seconds. Secrets, authorization headers, and API keys "
        "are redacted before events enter memory or disk."
    )
    download_col, app_log_col = st.columns(2)
    download_col.download_button(
        "Download recent traces (JSONL)",
        data=export_recent_traces(limit=2000),
        file_name="agent-traces.jsonl",
        mime="application/x-ndjson",
        key="download_agent_traces",
        width="stretch",
    )
    app_tail = read_log_tail(settings.logging.file, max_lines=1000)
    app_log_col.download_button(
        "Download application log tail",
        data=app_tail.encode("utf-8"),
        file_name="ai-analyst-log-tail.txt",
        mime="text/plain",
        key="download_application_log",
        width="stretch",
    )
    st.code(
        f"Application log: {settings.logging.file}\n"
        f"Structured trace log: {settings.logging.trace_file}",
        language=None,
    )


def _option_index(options: list, preferred) -> int:
    return options.index(preferred) if preferred in options else 0


def _render_chart_workspace(
    response: AnalysisResponse,
    response_key: str,
    *,
    selected_provider: str,
    selected_model: str | None,
    llm_ready: bool,
) -> None:
    dataframe = response.dataframe
    capabilities = inspect_chart_options(dataframe)
    graph_tab, followup_tab = st.tabs(["Build a graph", "Ask a follow-up"])

    with graph_tab:
        if not capabilities.applicable:
            _render_actionable_issue(
                "A graph is not applicable to this result",
                capabilities.reason,
                capabilities.suggestions,
                level="info",
            )
        else:
            st.success(capabilities.reason)
            st.markdown("###### AI visualization")
            st.caption(
                "Use the general recommendation, or describe exactly how the selected "
                "LLM should visualize these retrieved rows."
            )

            if st.button(
                "Visualize retrieved data",
                type="primary",
                key=f"recommended_chart::{response_key}",
                disabled=st.session_state.busy,
            ):
                visible = set(st.session_state.visible_recommended_charts)
                visible.add(response_key)
                st.session_state.visible_recommended_charts = visible

            if response_key in st.session_state.visible_recommended_charts:
                if response.chart is not None:
                    st.plotly_chart(
                        response.chart,
                        width="stretch",
                        key=f"recommended_plot::{response_key}",
                    )
                    st.caption(
                        "General graph recommended by the selected AI from the original question."
                    )
                else:
                    _render_actionable_issue(
                        "No useful automatic graph was found",
                        "The original result or AI plan did not contain a valid chart mapping.",
                        (
                            "Describe the exact chart and axes in AI visualization.",
                            "Use the manual controls with one of the applicable chart types.",
                            "Ask a follow-up that includes a category/date and numeric measure.",
                        ),
                        level="warning",
                    )

            with st.form(key=f"ai_chart_form::{response_key}", clear_on_submit=False):
                ai_chart_request = st.text_area(
                    "Describe the graph you need",
                    placeholder=(
                        "e.g. Plot the retrieved numeric measure by month, split by an "
                        "available category"
                    ),
                    key=f"ai_chart_request::{response_key}",
                    disabled=st.session_state.busy or not llm_ready,
                )
                ai_chart_clicked = st.form_submit_button(
                    "Generate with AI",
                    disabled=st.session_state.busy or not llm_ready,
                )

            ai_result_key = f"ai::{response_key}"
            if ai_chart_clicked:
                with st.spinner(operation_status(
                    selected_provider,
                    selected_model,
                    "chart",
                )):
                    st.session_state.generated_charts[ai_result_key] = generate_ai_result_chart(
                        dataframe,
                        request=ai_chart_request,
                        llm_provider=selected_provider,
                        llm_model=selected_model,
                    )

            ai_result = st.session_state.generated_charts.get(ai_result_key)
            if ai_result is not None:
                if ai_result.ok:
                    st.plotly_chart(
                        ai_result.figure,
                        width="stretch",
                        key=f"ai_plot::{response_key}",
                    )
                    if ai_result.ai_plan is not None:
                        st.caption(f"AI plan: {ai_result.ai_plan.explanation}")
                    st.caption(f"Graph generated from {ai_result.rows_plotted:,} plotted row(s).")
                else:
                    _render_actionable_issue(
                        ai_result.error_title or "The AI graph could not be generated",
                        ai_result.error or "The graph request could not be completed.",
                        ai_result.suggestions,
                    )

            st.divider()
            st.markdown("###### Manual graph controls")
            st.caption("Use these controls when you want exact axes and aggregation without an AI call.")
            chart_type = st.selectbox(
                "Chart type",
                list(capabilities.chart_types),
                format_func=lambda value: value.title(),
                key=f"chart_type::{response_key}",
            )

            if chart_type in {"histogram", "scatter"}:
                x_options: list[str | None] = list(capabilities.numeric_columns)
            elif chart_type == "box":
                x_options = [None, *capabilities.color_columns, *capabilities.datetime_columns]
            else:
                x_options = list(capabilities.x_columns)

            default_x = capabilities.default_x
            if chart_type in {"scatter", "histogram"} and default_x not in x_options:
                default_x = x_options[0]
            if chart_type == "box" and default_x not in x_options:
                default_x = None

            control_a, control_b, control_c = st.columns(3)
            with control_a:
                x = st.selectbox(
                    "X axis",
                    x_options,
                    index=_option_index(x_options, default_x),
                    format_func=lambda value: "No category" if value is None else str(value),
                    key=f"chart_x::{response_key}::{chart_type}",
                )

            if chart_type == "histogram":
                y = None
            else:
                y_options: list[str | None] = [None, *capabilities.numeric_columns]
                default_y = capabilities.default_y
                if chart_type == "scatter":
                    remaining = [column for column in capabilities.numeric_columns if column != x]
                    default_y = default_y if default_y in remaining else (remaining[0] if remaining else None)
                    y_options = list(capabilities.numeric_columns)
                with control_b:
                    y = st.selectbox(
                        "Y axis",
                        y_options,
                        index=_option_index(y_options, default_y),
                        format_func=lambda value: "Count rows" if value is None else str(value),
                        key=f"chart_y::{response_key}::{chart_type}",
                    )

            color_options: list[str | None] = [None, *capabilities.color_columns]
            with control_c:
                color = st.selectbox(
                    "Group / color",
                    color_options,
                    format_func=lambda value: "None" if value is None else str(value),
                    key=f"chart_color::{response_key}::{chart_type}",
                )

            settings_a, settings_b = st.columns(2)
            if chart_type in {"histogram", "scatter", "box"}:
                aggregation = "none"
                with settings_a:
                    st.caption("Aggregation is not used for this chart type.")
            else:
                aggregation_options = ["count"] if y is None else [
                    "sum", "avg", "count", "min", "max", "none"
                ]
                preferred_aggregation = capabilities.default_aggregation
                if preferred_aggregation not in aggregation_options:
                    preferred_aggregation = aggregation_options[0]
                with settings_a:
                    aggregation = st.selectbox(
                        "Aggregation",
                        aggregation_options,
                        index=_option_index(aggregation_options, preferred_aggregation),
                        format_func=lambda value: {
                            "avg": "Average",
                            "none": "No aggregation",
                        }.get(value, value.title()),
                        key=f"chart_aggregation::{response_key}::{chart_type}",
                    )

            can_group_time = (
                x in capabilities.datetime_columns
                and chart_type not in {"histogram", "scatter", "box"}
            )
            with settings_b:
                if can_group_time:
                    time_grain = st.selectbox(
                        "Time grouping",
                        ["none", "day", "week", "month", "quarter", "year"],
                        format_func=lambda value: "Original dates" if value == "none" else value.title(),
                        key=f"chart_time::{response_key}::{chart_type}",
                    )
                else:
                    time_grain = "none"
                    if capabilities.datetime_columns:
                        st.caption("Select a date column on the X axis to enable weekly/monthly grouping.")

            title = st.text_input(
                "Chart title",
                value=response.question[:100],
                key=f"chart_title::{response_key}::{chart_type}",
            )
            if st.button(
                "Generate graph",
                type="primary",
                key=f"generate_chart::{response_key}",
                disabled=st.session_state.busy,
            ):
                with st.spinner("Building graph from the retrieved data..."):
                    st.session_state.generated_charts[f"manual::{response_key}"] = generate_result_chart(
                        dataframe,
                        chart_type=chart_type,
                        x=x,
                        y=y,
                        color=color,
                        aggregation=aggregation,
                        time_grain=time_grain,
                        title=title,
                    )

            chart_result = st.session_state.generated_charts.get(f"manual::{response_key}")
            if chart_result is not None:
                if chart_result.ok:
                    st.plotly_chart(
                        chart_result.figure,
                        width="stretch",
                        key=f"exploratory_chart::{response_key}",
                    )
                    st.caption(f"Graph generated from {chart_result.rows_plotted:,} plotted row(s).")
                else:
                    _render_actionable_issue(
                        chart_result.error_title or "The graph could not be generated",
                        chart_result.error or "The selected options are not applicable.",
                        chart_result.suggestions,
                    )

            if not capabilities.datetime_columns:
                st.caption(
                    "Weekly/monthly grouping needs a date column in this result. Use the "
                    "follow-up tab to ask for the same measure broken down by week or month."
                )

    with followup_tab:
        st.caption(
            "Ask another question using this result's question, SQL, columns, and row count "
            "as context. A new read-only query will be generated when more detail is needed."
        )
        with st.form(key=f"followup_form::{response_key}", clear_on_submit=True):
            followup = st.text_input(
                "Follow-up question",
                placeholder="e.g. Show the same metric monthly, then compare by category",
                key=f"followup_input::{response_key}",
                disabled=st.session_state.busy or not llm_ready,
            )
            followup_clicked = st.form_submit_button(
                "Ask follow-up",
                type="primary",
                disabled=st.session_state.busy or not llm_ready,
            )
        if followup_clicked:
            _request_question(
                followup,
                selected_provider,
                selected_model,
                context_response=response,
            )


def _render_response(
    response: AnalysisResponse,
    *,
    selected_provider: str,
    selected_model: str | None,
    llm_ready: bool,
) -> None:
    st.markdown(f'<div class="ai-question-card"><strong>{response.question}</strong></div>', unsafe_allow_html=True)
    _render_badges(response)

    if response.status == "error":
        _render_actionable_issue(
            response.error_title or "The question could not be answered",
            response.error or "An unknown error occurred.",
            response.error_suggestions,
        )
        if response.sql:
            with st.expander("Technical details: last SQL attempt"):
                st.code(response.sql, language="sql")
        _render_tool_activity(response.tool_records, label="Query tools")
        return

    if response.insight_summary:
        st.info(response.insight_summary)
    if response.time_context and response.time_context.get("applied"):
        st.caption(
            "Resolved time period: "
            + str(response.time_context.get("label") or "shared database period")
        )
        if response.time_context.get("coverage_note"):
            st.warning(str(response.time_context["coverage_note"]))
    if response.insight_findings:
        for finding in response.insight_findings:
            st.markdown(f"- {finding}")

    response_key = _response_key(response)
    col_data, col_sql = st.columns([3, 2])
    with col_data:
        st.markdown("##### 📋 Retrieved data")
        if response.dataframe is not None:
            preview_limit = max(1, get_settings().ui.preview_rows)
            display_dataframe = response.dataframe.head(preview_limit)
            if response.truncated:
                st.caption(
                    f"Showing {len(display_dataframe):,} row(s); the analysis window "
                    f"contains the first {response.row_count:,}. The complete export "
                    "is prepared only when requested."
                )
            else:
                st.caption(
                    f"Showing {len(display_dataframe):,} of {response.row_count:,} row(s)."
                )
            st.dataframe(display_dataframe, width="stretch", height=360)

            if response.dataframe.empty:
                _render_actionable_issue(
                    "The query returned no rows",
                    "The query ran successfully, but no records matched the requested conditions.",
                    (
                        "Widen or remove date/category filters.",
                        "Check the Available data section for valid values and date coverage.",
                        "Ask a less restrictive follow-up, then refine the result.",
                    ),
                    level="warning",
                )

            prepared_download = st.session_state.prepared_downloads.get(response_key)
            if prepared_download is None and response.download_sql:
                if st.button(
                    "Prepare complete CSV",
                    key=f"prepare_download::{response_key}",
                    width="stretch",
                    help=(
                        "Runs the validated export query on demand. This keeps normal "
                        "question answering fast even when the database has many rows."
                    ),
                ):
                    with st.spinner("Preparing the complete CSV..."):
                        prepared_download = prepare_complete_download(response)
                    st.session_state.prepared_downloads[response_key] = prepared_download

            if prepared_download is not None and prepared_download.status == "ok":
                label = (
                    "Download CSV (safety-capped)"
                    if prepared_download.truncated
                    else "Download complete CSV"
                )
                st.download_button(
                    label,
                    data=prepared_download.csv_data or b"",
                    file_name=f"analysis-{response_key}.csv",
                    mime="text/csv",
                    key=f"download::{response_key}",
                    width="stretch",
                )
                st.caption(f"CSV ready: {prepared_download.row_count:,} row(s).")
                if prepared_download.truncated:
                    _render_actionable_issue(
                        "The CSV reached its safety cap",
                        f"The download contains the first {prepared_download.row_count:,} rows.",
                        (
                            "Narrow the question with a date range or business filter.",
                            "Increase SQL_DOWNLOAD_MAX_ROWS in .env if a larger export is expected.",
                            "Restart Streamlit after changing the export limit.",
                        ),
                        level="warning",
                    )
            elif prepared_download is not None:
                _render_actionable_issue(
                    prepared_download.error_title or "The CSV could not be prepared",
                    prepared_download.error or "An unknown download error occurred.",
                    prepared_download.error_suggestions,
                    level="warning",
                )
    with col_sql:
        st.markdown("##### 🧾 Generated SQL")
        st.code(
            response.sql or "",
            language="sql",
            line_numbers=True,
            wrap_lines=False,
            height=360,
            width="stretch",
        )
        if response.sql_explanation:
            st.caption(response.sql_explanation)

    st.markdown("##### Further analysis")
    _render_chart_workspace(
        response,
        response_key,
        selected_provider=selected_provider,
        selected_model=selected_model,
        llm_ready=llm_ready,
    )
    _render_tool_activity(response.tool_records, label="Query tools")


def _render_knowledge_response(response: KnowledgeAnswerResponse) -> None:
    st.markdown(f"**{response.question}**")
    provider_label = provider_display_label(response.llm_provider)
    retrieval_label = (
        "MCP + document RAG · semantic"
        if response.retrieval_mode == "vector"
        else "MCP + document RAG · lexical"
    )
    status_badges = [
        f'<span class="ai-badge ai-badge-cache">{retrieval_label}</span>',
    ]
    if response.llm_model:
        status_badges.append(
            f'<span class="ai-badge ai-badge-time">{provider_label} · '
            f'{response.llm_model}</span>'
        )
    if response.cache_hit:
        status_badges.append(
            '<span class="ai-badge ai-badge-cache">↻ served from session cache</span>'
        )
    elif response.status == "ok":
        status_badges.append(
            f'<span class="ai-badge ai-badge-live">answered in '
            f'{response.elapsed_seconds:.1f}s</span>'
        )
    if response.trace_id:
        status_badges.append(
            f'<span class="ai-badge ai-badge-time">trace {response.trace_id}</span>'
        )
    st.markdown("".join(status_badges), unsafe_allow_html=True)

    if response.status == "error":
        _render_actionable_issue(
            response.error_title or "The knowledge question could not be answered",
            response.error or "An unknown error occurred.",
            response.error_suggestions,
        )
        _render_tool_activity(response.tool_records, label="Knowledge tools")
        return

    st.markdown(response.answer or "No answer was returned.")
    st.caption(
        f"Grounded in {len(response.sources)} retrieved document(s). "
        "Open a source below to verify the answer."
    )
    for source in response.sources:
        with st.expander(
            f"Source: {source.table_name} · knowledge base v{source.version}"
        ):
            st.code(source.content, language=None)
    _render_tool_activity(response.tool_records, label="Knowledge tools")


def _render_file_response(response: FileAssistantResponse) -> None:
    st.markdown(f"**{response.question}**")
    if response.llm_model:
        st.caption(
            f"{response.llm_provider or 'LLM'} · {response.llm_model} · "
            f"{len(response.tool_records)} MCP tool call(s)"
        )
    if response.trace_id:
        st.caption(f"Trace: `{response.trace_id}`")
    if response.status != "ok":
        _render_actionable_issue(
            "Filesystem request could not be completed",
            response.error or "The MCP tool loop failed without an error message.",
            (
                "Confirm Node.js, npm, and the Python MCP package are installed.",
                "Check that every requested path is under an allowed MCP root.",
                "Use Azure Kimi K2.6, NVIDIA GLM 5.2, or MiniMax M3 if another hosted model is degraded.",
            ),
        )
    else:
        st.markdown(response.answer or "The model returned no final explanation.")
    if response.tool_records:
        with st.expander("Filesystem actions and results", expanded=False):
            for index, record in enumerate(response.tool_records, start=1):
                status = "failed" if record.is_error else "completed"
                st.markdown(f"**{index}. `{record.name}` — {status}**")
                st.json(record.arguments)
                st.code(record.result, language=None)


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()
    ollama_status = _start_required_services()
    selected_provider, selected_model, llm_ready = _render_sidebar(ollama_status)

    st.title("Ask your data & docs")
    st.caption(
        "Query live data through MCP + schema RAG and validated read-only SQL, "
        "or ask grounded questions retrieved from the knowledge base through MCP."
    )

    data_tab, knowledge_tab, files_tab, diagnostics_tab = st.tabs(
        ["Query data", "Ask knowledge base", "Work with files (MCP)", "Live agent logs"]
    )

    with data_tab:
        st.caption(
            "Ask for totals, trends, rankings, comparisons, or records. An MCP tool "
            "RAG-selects schema context, then MCP validates and executes the generated "
            "SQL read-only."
        )
        with st.form(key="ask_form", clear_on_submit=False):
            question = st.text_input(
                "Your data question",
                key="question_input",
                placeholder="e.g. What was the documented metric by category last year?",
                label_visibility="collapsed",
                disabled=st.session_state.busy or not llm_ready,
            )
            ask_clicked = st.form_submit_button(
                "Query data",
                type="primary",
                disabled=st.session_state.busy or not llm_ready,
            )

        if st.session_state.busy and st.session_state.pending_question:
            st.caption(f"Working on: *{st.session_state.pending_question}*")
            _run_pending_question()
        elif ask_clicked:
            _request_question(question, selected_provider, selected_model)

        st.divider()
        if not st.session_state.history and not st.session_state.busy:
            st.caption("No data questions asked yet — try an example from the sidebar.")

        for i, response in enumerate(st.session_state.history):
            _render_response(
                response,
                selected_provider=selected_provider,
                selected_model=selected_model,
                llm_ready=llm_ready,
            )
            if i < len(st.session_state.history) - 1:
                st.divider()

    with knowledge_tab:
        st.caption(
            "Ask about tables, columns, relationships, categories, metric definitions, "
            "or aggregation guidance. An MCP tool retrieves relevant RAG documents first "
            "and supplies them to the selected LLM; this mode does not execute SQL."
        )
        with st.form(key="knowledge_ask_form", clear_on_submit=False):
            knowledge_question = st.text_input(
                "Your knowledge-base question",
                key="knowledge_question_input",
                placeholder="e.g. Which tables are related, and what do their columns mean?",
                label_visibility="collapsed",
                disabled=st.session_state.busy or not llm_ready,
            )
            knowledge_clicked = st.form_submit_button(
                "Ask documentation",
                type="primary",
                disabled=st.session_state.busy or not llm_ready,
            )
        st.caption(
            "Try: “How are the main tables related?” or "
            "“Which numeric columns can be aggregated?”"
        )

        if st.session_state.busy and st.session_state.pending_knowledge_question:
            st.caption(
                f"Retrieving sources for: *{st.session_state.pending_knowledge_question}*"
            )
            _run_pending_knowledge_question()
        elif knowledge_clicked:
            _request_knowledge_question(
                knowledge_question,
                selected_provider,
                selected_model,
            )

        st.divider()
        if not st.session_state.knowledge_history and not st.session_state.busy:
            st.caption("No documentation questions asked yet.")
        for i, response in enumerate(st.session_state.knowledge_history):
            _render_knowledge_response(response)
            if i < len(st.session_state.knowledge_history) - 1:
                st.divider()

    with files_tab:
        filesystem = get_settings().mcp_filesystem
        roots = [str(path) for path in filesystem.roots]
        tool_capable = bool(
            selected_model
            and (
                (
                    selected_provider == "azure_foundry"
                    and "kimi-k2.6" in selected_model.casefold()
                )
                or (
                    selected_provider == "nvidia_nim"
                    and selected_model
                    in {
                        "nvidia/nemotron-3-ultra-550b-a55b",
                        "poolside/laguna-xs-2.1",
                        "z-ai/glm-5.2",
                        "minimaxai/minimax-m3",
                    }
                )
            )
        )
        st.caption(
            "Ask the selected model to inspect files through the official "
            "@modelcontextprotocol/server-filesystem server. Paths are sandboxed "
            "to the configured roots. Delete operations are never exposed."
        )
        st.markdown("**Allowed roots**")
        st.code("\n".join(roots) if roots else "No roots configured", language=None)
        if not filesystem.is_configured:
            _render_actionable_issue(
                "Filesystem MCP is disabled",
                "Enable MCP_FILESYSTEM_ENABLED and configure at least one root.",
            )
        elif not tool_capable:
            _render_actionable_issue(
                "Select a tool-capable model",
                f"{provider_display_label(selected_provider)} · "
                f"{selected_model or 'no model'} is not enabled for the app's "
                "filesystem tool loop. OpenRouter and Ollama remain available for "
                "data and knowledge-base questions.",
                ("Select Azure AI Foundry · Kimi-K2.6 for the verified working path.",),
                level="warning",
            )

        with st.form(key="filesystem_mcp_form", clear_on_submit=False):
            file_question = st.text_area(
                "File question or action",
                key="file_question_input",
                placeholder=(
                    "e.g. Find README files under the project and summarize the main one."
                ),
                disabled=st.session_state.busy or not filesystem.is_configured,
            )
            request_mutations = st.checkbox(
                "Allow create, write, edit, and move tools for this request",
                value=False,
                disabled=(
                    st.session_state.busy
                    or not filesystem.allow_mutations
                    or not filesystem.is_configured
                ),
                help=(
                    "Requires MCP_FILESYSTEM_ALLOW_MUTATIONS=true. Delete tools remain blocked."
                ),
            )
            mutation_confirmation = st.checkbox(
                "I approve file changes inside the displayed roots",
                value=False,
                disabled=st.session_state.busy or not request_mutations,
            )
            file_clicked = st.form_submit_button(
                "Run filesystem request",
                type="primary",
                disabled=(
                    st.session_state.busy
                    or not llm_ready
                    or not tool_capable
                    or not filesystem.is_configured
                    or (request_mutations and not mutation_confirmation)
                ),
            )
        if not filesystem.allow_mutations:
            st.info(
                "Read-only mode is active. Set MCP_FILESYSTEM_ALLOW_MUTATIONS=true "
                "in .env and restart to make the per-request write approval available."
            )

        if st.session_state.busy and st.session_state.pending_file_question:
            st.caption(
                f"Using filesystem tools for: *{st.session_state.pending_file_question}*"
            )
            _run_pending_file_question()
        elif file_clicked:
            _request_file_question(
                file_question,
                selected_provider,
                selected_model,
                allow_mutations=request_mutations and mutation_confirmation,
            )

        st.divider()
        if not st.session_state.file_history and not st.session_state.busy:
            st.caption("No filesystem MCP requests have been made yet.")
        for i, response in enumerate(st.session_state.file_history):
            _render_file_response(response)
            if i < len(st.session_state.file_history) - 1:
                st.divider()

    with diagnostics_tab:
        st.caption(
            "Follow each request across the agent workflow, LLM/provider calls, "
            "database tools, SQL validation/execution, and filesystem MCP tools."
        )
        _render_live_diagnostics()


if __name__ == "__main__":
    main()
