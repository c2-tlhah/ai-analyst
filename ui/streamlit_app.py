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
from app.orchestrator import (
    AnalysisResponse,
    answer_question,
    clear_session_caches,
    connect_database,
    generate_ai_result_chart,
    generate_result_chart,
    get_active_database_info,
    get_error_guidance,
    get_knowledge_base_documents,
    get_llm_catalog,
    get_session_stats,
    get_table_catalog,
    inspect_chart_options,
    list_knowledge_base_versions,
    refresh_metadata,
)
from app.services import ServiceStatus, ensure_ollama_running

st.set_page_config(page_title="AI Analyst", page_icon="🌱", layout="wide")

EXAMPLE_QUESTIONS = [
    "What are the top 10 products by total sales amount across both channels?",
    "Compare monthly internet sales revenue over time.",
    "Which product line has the highest average discount percentage?",
    "How many orders did each sales territory generate through the reseller channel?",
    "What is the distribution of list prices for finished-goods products?",
]

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
    if "generated_charts" not in st.session_state:
        st.session_state.generated_charts = {}
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
        with st.spinner("Thinking through your question... (reasoning models can take up to a minute)"):
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


def _render_sidebar(ollama_status: ServiceStatus) -> tuple[str, str | None, bool]:
    settings = get_settings()
    provider_before_render = st.session_state.get(
        "llm_provider", settings.default_llm_provider
    )
    llm_catalog = get_llm_catalog(
        discover_ollama=provider_before_render == "ollama"
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

        selected_model: str | None = None
        if models:
            model_key = f"llm_model::{selected_provider}"
            configured_model = (
                {
                    "ollama": settings.ollama.model,
                    "azure_foundry": settings.azure_ai.model_deployment,
                    "openrouter": settings.openrouter.model,
                }.get(selected_provider, "")
            )
            if st.session_state.get(model_key) not in models:
                st.session_state[model_key] = (
                    configured_model if configured_model in models else models[0]
                )
            selected_model = st.selectbox(
                "Model",
                models,
                key=model_key,
                disabled=busy,
            )

        llm_ready = bool(provider_info["available"] and selected_model)
        if llm_ready:
            st.success(f"Ready: {selected_model}")
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

        db_info = get_active_database_info()
        if db_info["vector_indexed"]:
            st.caption(
                f"🔎 Knowledge base: {db_info['vector_table_count']} table(s) indexed -- "
                f"version {db_info['vector_version']} of {db_info['vector_version_count']} "
                "(one version per schema change; earlier versions are kept, not overwritten)."
            )
            if st.checkbox("📄 View knowledge base documents", key="kb_show_docs", disabled=busy):
                versions = list_knowledge_base_versions()
                version_options = [v["version"] for v in versions][::-1]  # latest first
                if st.session_state.get("kb_version_select") not in version_options:
                    st.session_state.kb_version_select = version_options[0]
                version_by_number = {v["version"]: v for v in versions}
                chosen_version = st.selectbox(
                    "Version",
                    version_options,
                    format_func=lambda v: (
                        f"v{v} -- {version_by_number[v]['created_at'][:19]} "
                        f"({len(version_by_number[v]['tables'])} table(s))"
                        + (" (current)" if v == db_info["vector_version"] else "")
                    ),
                    key="kb_version_select",
                )
                st.caption(
                    "Saved as plain text under "
                    f"`vector_store/knowledge_base_txt/.../v{chosen_version}/`."
                )
                for table_name, text in get_knowledge_base_documents(chosen_version).items():
                    with st.expander(table_name):
                        st.code(text, language=None)
        else:
            st.caption("🔎 Knowledge base: not indexed yet -- connect to build it.")

        with st.form(key="db_connect_form", clear_on_submit=False):
            st.text_input(
                "SQLite file path or connection string",
                key="db_source_input",
                placeholder="e.g. data/ai_analyst.db or sqlite:///data/ai_analyst.db",
                disabled=busy,
            )
            connect_clicked = st.form_submit_button(
                "Connect & build knowledge base",
                disabled=busy or not llm_ready,
            )
        if connect_clicked:
            with st.spinner(
                "Crawling schema, generating descriptions, and building the "
                "vector knowledge base..."
            ):
                connect_result = connect_database(
                    st.session_state.db_source_input,
                    llm_provider=selected_provider,
                    llm_model=selected_model,
                )
            st.session_state.db_connect_result = connect_result
            if connect_result.success and connect_result.db_path:
                st.session_state._db_source_pending = connect_result.db_path
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
            for table in catalog:
                with st.expander(f"{table['name']}  ·  {table['kind']}"):
                    st.caption(table["description"])
                    st.caption(f"~{table['row_count']:,} rows")
        except Exception as exc:  # noqa: BLE001
            guidance = get_error_guidance(exc, stage="database")
            _render_actionable_issue(
                "The data catalog could not be loaded",
                guidance.reason,
                (
                    "Refresh the schema from the sidebar.",
                    "Verify the database path and run scripts/build_database.py if needed.",
                    "Check logs/ai_analyst.log if the problem repeats.",
                ),
                level="warning",
            )

        st.divider()
        st.subheader("Try asking")
        for q in EXAMPLE_QUESTIONS:
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
            st.session_state.generated_charts = {}
            st.session_state.visible_recommended_charts = set()
            st.rerun()

    return selected_provider, selected_model, llm_ready


def _render_badges(response: AnalysisResponse) -> None:
    badges = []
    if response.llm_model:
        provider_label = {
            "azure_foundry": "Azure AI Foundry",
            "ollama": "Ollama",
            "openrouter": "OpenRouter",
        }.get(response.llm_provider or "", response.llm_provider or "LLM")
        badges.append(
            f'<span class="ai-badge ai-badge-time">{provider_label} · {response.llm_model}</span>'
        )
    if response.retrieval_mode == "vector":
        badges.append('<span class="ai-badge ai-badge-cache">🔎 RAG-retrieved schema context</span>')
    elif response.retrieval_mode == "lexical":
        badges.append('<span class="ai-badge ai-badge-time">🔤 keyword-matched schema context</span>')
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
    st.markdown("".join(badges), unsafe_allow_html=True)


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
                        "e.g. Show total sales by month as a line chart, with a separate "
                        "series for each sales channel"
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
                with st.spinner("The selected AI is planning and validating your graph..."):
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
                placeholder="e.g. Show the same sales monthly, then compare by channel",
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
        return

    if response.insight_summary:
        st.info(response.insight_summary)
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
            total_available = response.download_row_count or response.row_count
            st.caption(
                f"Showing {len(display_dataframe):,} of {total_available:,} row(s)."
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

            if response.download_dataframe is not None:
                label = (
                    "Download CSV (safety-capped)"
                    if response.download_truncated
                    else "Download complete CSV"
                )
                st.download_button(
                    label,
                    data=response.download_dataframe.to_csv(index=False).encode("utf-8"),
                    file_name=f"analysis-{response_key}.csv",
                    mime="text/csv",
                    key=f"download::{response_key}",
                    width="stretch",
                )
                if response.download_truncated:
                    _render_actionable_issue(
                        "The CSV reached its safety cap",
                        f"The download contains the first {response.download_row_count:,} rows.",
                        (
                            "Narrow the question with a date range or business filter.",
                            "Increase SQL_DOWNLOAD_MAX_ROWS in .env if a larger export is expected.",
                            "Restart Streamlit after changing the export limit.",
                        ),
                        level="warning",
                    )
            elif response.download_error:
                guidance = get_error_guidance(response.download_error, stage="download")
                _render_actionable_issue(
                    guidance.title,
                    guidance.reason,
                    guidance.suggestions,
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


def main() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    _init_state()
    ollama_status = _start_required_services()
    selected_provider, selected_model, llm_ready = _render_sidebar(ollama_status)

    st.title("Ask your data")
    st.caption(
        "Ask a question in plain English. The system plans the query, validates and "
        "runs it read-only, then summarizes and visualizes the result."
    )

    with st.form(key="ask_form", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            key="question_input",
            placeholder="e.g. What were total sales by product line last year?",
            label_visibility="collapsed",
            disabled=st.session_state.busy or not llm_ready,
        )
        ask_clicked = st.form_submit_button(
            "Ask",
            type="primary",
            disabled=st.session_state.busy or not llm_ready,
        )

    if st.session_state.busy and st.session_state.pending_question:
        st.caption(f"Working on: *{st.session_state.pending_question}*")
        _run_pending_question()
    elif ask_clicked:
        _request_question(question, selected_provider, selected_model)

    st.divider()

    if not st.session_state.history:
        st.caption("No questions asked yet -- try one of the examples in the sidebar.")

    for i, response in enumerate(st.session_state.history):
        _render_response(
            response,
            selected_provider=selected_provider,
            selected_model=selected_model,
            llm_ready=llm_ready,
        )
        if i < len(st.session_state.history) - 1:
            st.divider()


if __name__ == "__main__":
    main()
