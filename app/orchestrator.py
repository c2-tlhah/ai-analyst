"""High-level entrypoint the Streamlit UI (and tests) call into.

Streamlit never touches the database, the LLM, or LangGraph directly -- it
only calls :func:`answer_question` and renders the :class:`AnalysisResponse`
it gets back. This module is the seam that keeps the UI purely
presentational, per the "Streamlit only handles UI" requirement.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import pandas as pd
import plotly.graph_objects as go

from app.config import get_settings
from app.db.connection import (
    DatabaseNotFoundError,
    InvalidDatabaseSourceError,
    get_active_database_identity,
    get_active_database_path,
    readonly_connection,
    set_active_database_path,
    validate_database_source,
)
from app.db.executor import execute_sql
from app.error_guidance import explain_error
from app.graph.workflow import build_workflow
from app.llm.client import (
    AZURE_FOUNDRY_PROVIDER,
    OLLAMA_PROVIDER,
    OPENROUTER_PROVIDER,
    LLMClient,
    LLMError,
    get_llm_client,
    get_usage_stats,
    list_ollama_models,
)
from app.logging_config import get_logger
from app.metadata import enrichment, store, vector_store
from app.viz.explorer import (
    ChartBuildResult,
    ChartCapabilities,
    build_exploratory_chart,
    generate_ai_exploratory_chart,
    get_chart_capabilities,
)

logger = get_logger(__name__)


@dataclass
class AnalysisResponse:
    status: str
    question: str
    sql: Optional[str] = None
    sql_explanation: Optional[str] = None
    dataframe: Optional[pd.DataFrame] = None
    download_dataframe: Optional[pd.DataFrame] = None
    row_count: int = 0
    download_row_count: int = 0
    truncated: bool = False
    download_truncated: bool = False
    download_error: Optional[str] = None
    insight_summary: Optional[str] = None
    insight_findings: list[str] = field(default_factory=list)
    chart: Optional[go.Figure] = None
    error: Optional[str] = None
    error_title: Optional[str] = None
    error_stage: Optional[str] = None
    error_suggestions: tuple[str, ...] = ()
    retry_count: int = 0
    # Session-memory/caching metadata -- purely informational for the UI.
    cache_hit: bool = False
    elapsed_seconds: float = 0.0
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    # "vector" (RAG similarity search) or "lexical" (keyword fallback) --
    # which strategy in app.metadata.retrieval picked the relevant tables.
    retrieval_mode: Optional[str] = None


_workflow_cache: dict[int, Any] = {}


def _get_compiled_workflow(llm_client: LLMClient):
    key = id(llm_client)
    if key not in _workflow_cache:
        _workflow_cache[key] = build_workflow(llm_client)
    return _workflow_cache[key]


# ---------------------------------------------------------------------------
# Metadata "session memory": re-hashing the live schema (PRAGMA table_info /
# foreign_key_list / sample-value queries for every table) is cheap once, but
# pointless to repeat on every single question within the same run -- the
# schema essentially never changes mid-session. Cache the result in-process
# and only re-verify against the database after it goes stale, or when the
# caller explicitly forces a check (e.g. an in-UI "Refresh schema" button).
# ---------------------------------------------------------------------------
_METADATA_CACHE_TTL_SECONDS = 300.0
_metadata_cache: dict[str, Any] = {"metadata": None, "checked_at": 0.0, "source": None}


def refresh_metadata(llm_client: Optional[LLMClient] = None, force: bool = False) -> dict[str, Any]:
    """Re-discover the schema if it drifted and return the current metadata.

    Cheap when nothing changed (a structural hash comparison); only pays for
    a full rebuild -- and only calls the LLM to describe genuinely new
    tables/columns -- when the live schema no longer matches the cache. On
    top of that, this in-process cache skips the hash check entirely for
    ``_METADATA_CACHE_TTL_SECONDS`` after the last verification, so a whole
    session of questions pays for schema discovery once, not once per
    question.
    """
    now = time.monotonic()
    is_fresh = (now - _metadata_cache["checked_at"]) < _METADATA_CACHE_TTL_SECONDS
    if not force and _metadata_cache["metadata"] is not None and is_fresh:
        _metadata_cache["source"] = "session_cache"
        return _metadata_cache["metadata"]

    enrich_fn = enrichment.make_llm_enrich_fn(llm_client) if llm_client else None
    with readonly_connection() as conn:
        metadata, was_rebuilt = store.refresh_if_needed(conn, enrich_fn=enrich_fn, force=force)
    if was_rebuilt:
        logger.info("Metadata store rebuilt (schema change detected).")
        if get_settings().vector.enabled:
            vector_store.sync_collection(metadata, db_identity=get_active_database_identity())

    _metadata_cache["metadata"] = metadata
    _metadata_cache["checked_at"] = now
    _metadata_cache["source"] = "rebuilt" if was_rebuilt else "verified"
    return metadata


@dataclass
class ConnectResult:
    success: bool
    message: str
    db_path: Optional[str] = None
    table_count: int = 0
    indexed_table_count: int = 0
    knowledge_base_version: int = 0
    knowledge_base_status: str = "not_built"
    knowledge_base_error: Optional[str] = None


def connect_database(
    source: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> ConnectResult:
    """Point the app at a different SQLite database from the UI.

    Validates ``source`` (a filesystem path or ``sqlite:///`` connection
    string), makes it the active database, crawls its schema (deterministic
    discovery + LLM-assisted description of anything not already curated --
    see ``app.metadata.enrichment``), and (re)builds its vector knowledge
    base (``app.metadata.vector_store``) so RAG retrieval has something to
    search on the very first question against it. A structural schema change
    since the last time this database was indexed mints a new, numbered
    knowledge-base version rather than overwriting the previous one.
    """
    try:
        path = validate_database_source(source)
    except (DatabaseNotFoundError, InvalidDatabaseSourceError) as exc:
        return ConnectResult(success=False, message=str(exc))

    set_active_database_path(path)
    # A different database invalidates both process-wide caches: cached
    # metadata described a different schema, and cached answers were run
    # against different data.
    _metadata_cache["metadata"] = None
    _metadata_cache["checked_at"] = 0.0
    _answer_cache.clear()

    llm_client: Optional[LLMClient] = None
    if llm_provider:
        try:
            llm_client = get_llm_client(provider=llm_provider, model=llm_model)
        except Exception:  # noqa: BLE001 - schema crawling still works without an LLM
            logger.warning(
                "No LLM available for schema enrichment; new tables will get "
                "humanized-name descriptions instead of AI-generated ones."
            )

    try:
        metadata = refresh_metadata(llm_client, force=True)
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI, not raised
        logger.exception("Schema discovery failed for %s", path)
        return ConnectResult(
            success=False,
            message=f"Connected to {path.name}, but schema discovery failed: {exc}",
            db_path=str(path),
        )

    table_count = len(metadata.get("tables", {}))
    # refresh_metadata(force=True) above already synced the knowledge base
    # (it always rebuilds under force=True); read back what that produced
    # rather than re-embedding everything a second time here.
    kb_status = vector_store.collection_stats(get_active_database_identity())
    indexed = kb_status.get("document_count", 0) if kb_status.get("indexed") else 0
    version_number = kb_status.get("version", 0)
    status = kb_status.get("status", "not_built")
    if status == "ready":
        kb_note = f"{indexed} indexed into knowledge base version {version_number}."
    elif status == "disabled":
        kb_note = (
            "the knowledge base was not built because VECTOR_RAG_ENABLED=false. "
            "Set it to true and reconnect to enable semantic schema retrieval."
        )
    elif status == "error":
        kb_note = f"the knowledge-base build failed: {kb_status.get('error') or 'unknown error'}"
    else:
        kb_note = "the knowledge base has not been built yet."

    return ConnectResult(
        success=True,
        message=f"Connected to {path.name}: {table_count} table(s) discovered, {kb_note}",
        db_path=str(path),
        table_count=table_count,
        indexed_table_count=indexed,
        knowledge_base_version=version_number,
        knowledge_base_status=status,
        knowledge_base_error=kb_status.get("error"),
    )


def rebuild_active_knowledge_base(
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> ConnectResult:
    """Re-crawl and sync the active database's knowledge base on demand.

    LLM enrichment is optional: deterministic schema documents and vector
    indexing still work when no remote/local model is configured.
    """
    return connect_database(
        str(get_active_database_path()),
        llm_provider=llm_provider,
        llm_model=llm_model,
    )


def get_active_database_info() -> dict[str, Any]:
    """Active database path + vector-index status, for the UI connection panel."""
    path = get_active_database_path()
    identity = get_active_database_identity()
    stats = vector_store.collection_stats(identity)
    return {
        "path": str(path),
        "exists": path.exists(),
        "vector_indexed": stats["indexed"],
        "vector_table_count": stats["table_count"],
        "vector_version": stats.get("version", 0),
        "vector_version_count": stats.get("version_count", 0),
        "vector_enabled": stats.get("enabled", False),
        "vector_status": stats.get("status", "not_built"),
        "vector_error": stats.get("error"),
        "vector_document_count": stats.get("document_count", 0),
        "vector_text_export_path": stats.get("text_export_path"),
    }


def list_knowledge_base_versions() -> list[dict[str, Any]]:
    """Every knowledge-base version built for the active database, oldest first.

    Each schema change (a column/table added, changed, or removed) mints a
    new version rather than overwriting the last one -- this is what the
    UI's version picker lists.
    """
    return vector_store.list_versions(get_active_database_identity())


def get_knowledge_base_documents(version: Optional[int] = None) -> dict[str, str]:
    """The human-readable per-table documents for one knowledge-base version.

    Defaults to the latest version for the active database. Returns
    ``{table_name: document_text}``, the same text that's embedded for RAG
    retrieval and saved under ``vector_store/knowledge_base_txt/``.
    """
    identity = get_active_database_identity()
    if version is None:
        version = vector_store.collection_stats(identity).get("version", 0)
    if not version:
        return {}
    return vector_store.read_version_documents(identity, version)


def get_metadata_cache_info() -> dict[str, Any]:
    """Session-memory status for the UI (age, TTL, whether it was a live check)."""
    checked_at = _metadata_cache["checked_at"]
    age = time.monotonic() - checked_at if checked_at else None
    return {
        "cached": _metadata_cache["metadata"] is not None,
        "age_seconds": age,
        "ttl_seconds": _METADATA_CACHE_TTL_SECONDS,
        "source": _metadata_cache["source"],
    }


def get_table_catalog() -> list[dict[str, Any]]:
    """Lightweight table catalog for UI display (no LLM call, no disk I/O once cached)."""
    metadata = _metadata_cache["metadata"] or store.load_schema_metadata()
    if metadata is None:
        with readonly_connection() as conn:
            metadata, _ = store.refresh_if_needed(conn)
    return store.get_table_catalog(metadata)


def inspect_chart_options(dataframe: Optional[pd.DataFrame]) -> ChartCapabilities:
    """Return deterministic visualization options for an already retrieved result."""
    try:
        return get_chart_capabilities(dataframe)
    except Exception as exc:  # noqa: BLE001 - malformed data should not crash the UI
        logger.exception("Could not inspect graph options")
        return ChartCapabilities(
            applicable=False,
            reason=f"The retrieved columns could not be inspected safely: {exc}",
            suggestions=(
                "Run a simpler query with plain date, category, and numeric columns.",
                "Download the CSV to inspect the underlying values.",
                "Check logs/ai_analyst.log if the issue repeats.",
            ),
        )


def generate_result_chart(
    dataframe: Optional[pd.DataFrame],
    *,
    chart_type: str,
    x: str | None,
    y: str | None,
    color: str | None = None,
    aggregation: str = "none",
    time_grain: str = "none",
    title: str = "Retrieved data",
) -> ChartBuildResult:
    """Build a validated chart without rerunning SQL or calling an LLM."""
    try:
        return build_exploratory_chart(
            dataframe,
            chart_type=chart_type,
            x=x,
            y=y,
            color=color,
            aggregation=aggregation,
            time_grain=time_grain,
            title=title,
        )
    except Exception as exc:  # noqa: BLE001 - malformed values should not crash Streamlit
        logger.exception("Manual graph generation failed")
        return ChartBuildResult(
            error=f"The retrieved values could not be transformed into this graph: {exc}",
            error_title="Graph generation failed",
            suggestions=(
                "Try the recommended chart type and axes.",
                "Remove time grouping or group/color, then retry.",
                "Download the CSV to inspect null or malformed values.",
            ),
        )


def generate_ai_result_chart(
    dataframe: Optional[pd.DataFrame],
    *,
    request: str,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> ChartBuildResult:
    """Interpret a natural-language chart request with the selected LLM."""
    try:
        llm_client = get_llm_client(provider=llm_provider, model=llm_model)
        return generate_ai_exploratory_chart(llm_client, dataframe, request=request)
    except Exception as exc:  # noqa: BLE001 - provider/configuration errors are user-facing
        logger.exception("AI chart generation failed")
        guidance = explain_error(exc, stage="provider")
        return ChartBuildResult(
            error=guidance.reason,
            error_title="AI graph planning failed",
            suggestions=guidance.suggestions,
        )


def get_error_guidance(error: object, *, stage: str = "application"):
    """Expose consistent backend error guidance to presentation-only clients."""
    return explain_error(error, stage=stage)


# Ollama model discovery is a fast local call, but Streamlit reruns the whole
# script frequently. Keep the result briefly and expose an explicit refresh.
_OLLAMA_MODEL_CACHE_TTL_SECONDS = 10.0
_ollama_model_cache: dict[str, Any] = {
    "models": [],
    "error": None,
    "checked_at": 0.0,
}


def get_llm_catalog(
    refresh_ollama: bool = False,
    discover_ollama: bool = True,
) -> dict[str, dict[str, Any]]:
    """Describe selectable providers/models without constructing an LLM client."""
    settings = get_settings()
    now = time.monotonic()
    cache_is_fresh = (
        _ollama_model_cache["checked_at"]
        and now - _ollama_model_cache["checked_at"] < _OLLAMA_MODEL_CACHE_TTL_SECONDS
    )
    if refresh_ollama or (discover_ollama and not cache_is_fresh):
        try:
            _ollama_model_cache["models"] = list_ollama_models(settings.ollama)
            _ollama_model_cache["error"] = None
        except LLMError as exc:
            _ollama_model_cache["models"] = []
            _ollama_model_cache["error"] = str(exc)
        _ollama_model_cache["checked_at"] = now

    ollama_models = list(_ollama_model_cache["models"])
    ollama_error = _ollama_model_cache["error"]
    azure_error = None
    if not settings.azure_ai.is_configured:
        azure_error = (
            "Set AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY in .env."
        )
    if not ollama_error and not ollama_models:
        ollama_error = (
            "Ollama is running, but no downloaded models were found."
            if _ollama_model_cache["checked_at"]
            else "Select Ollama to discover downloaded models."
        )
    openrouter_error = None
    if not settings.openrouter.is_configured:
        openrouter_error = "Set OPENROUTER_API_KEY in .env."

    return {
        AZURE_FOUNDRY_PROVIDER: {
            "label": "Azure AI Foundry",
            "models": [settings.azure_ai.model_deployment],
            "available": settings.azure_ai.is_configured,
            "error": azure_error,
            "error_guidance": (
                explain_error(azure_error, stage="configuration") if azure_error else None
            ),
        },
        OLLAMA_PROVIDER: {
            "label": "Ollama (local)",
            "models": ollama_models,
            "available": bool(ollama_models) and not _ollama_model_cache["error"],
            "error": ollama_error,
            "error_guidance": (
                explain_error(ollama_error, stage="service") if ollama_error else None
            ),
        },
        OPENROUTER_PROVIDER: {
            "label": "OpenRouter",
            "models": [settings.openrouter.model],
            "available": settings.openrouter.is_configured,
            "error": openrouter_error,
            "error_guidance": (
                explain_error(openrouter_error, stage="configuration")
                if openrouter_error
                else None
            ),
        },
    }


# ---------------------------------------------------------------------------
# Exact-question answer cache: re-asking a question already answered this
# session (clicking the same example twice, revisiting a prior question)
# skips the DB round-trip and all four LLM calls entirely and returns the
# same validated result instantly. Small and bounded (LRU-ish) since results
# hold live DataFrames/Figures.
# ---------------------------------------------------------------------------
_ANSWER_CACHE_MAX_SIZE = 50
_answer_cache: "OrderedDict[str, AnalysisResponse]" = OrderedDict()


def _cache_key(question: str, llm_namespace: str = "default") -> str:
    normalized_question = " ".join(question.strip().lower().split())
    return f"{llm_namespace}\n{normalized_question}"


def clear_session_caches() -> None:
    """Drop the answer cache (used by the UI's "New session" action)."""
    _answer_cache.clear()


def get_cache_stats() -> dict[str, int]:
    return {"cached_questions": len(_answer_cache)}


def answer_question(
    question: str,
    llm_client: Optional[LLMClient] = None,
    conversation_history: Optional[list[dict[str, Any]]] = None,
    use_cache: bool = True,
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
) -> AnalysisResponse:
    """Run the full LangGraph pipeline for a natural-language question.

    ``conversation_history`` is an optional, caller-managed list of recent
    ``{"question": ..., "sql": ...}`` turns from this session -- passed
    through to the intent/SQL-generation prompts so short follow-ups
    ("now break that down by year") can be resolved without repeating full
    context. ``use_cache`` short-circuits identical repeat questions.
    """
    settings = get_settings()
    question = (question or "").strip()
    if not question:
        guidance = explain_error("Please enter a question.", stage="input")
        return AnalysisResponse(
            status="error",
            question=question,
            error=guidance.reason,
            error_title=guidance.title,
            error_stage=guidance.stage,
            error_suggestions=guidance.suggestions,
        )

    try:
        llm_client = llm_client or get_llm_client(provider=llm_provider, model=llm_model)
    except Exception as exc:  # noqa: BLE001 - configuration/network errors are user-facing
        logger.warning("Could not initialize selected LLM: %s", exc)
        guidance = explain_error(exc, stage="configuration")
        return AnalysisResponse(
            status="error",
            question=question,
            error=guidance.reason,
            error_title=guidance.title,
            error_stage=guidance.stage,
            error_suggestions=guidance.suggestions,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )

    selected_provider = llm_client.provider_name
    selected_model = llm_client.model_name
    cache_key = _cache_key(question, llm_client.cache_namespace)
    if use_cache and cache_key in _answer_cache:
        cached = _answer_cache[cache_key]
        _answer_cache.move_to_end(cache_key)
        return replace(cached, cache_hit=True, elapsed_seconds=0.0)

    started = time.monotonic()
    try:
        metadata = refresh_metadata(llm_client)
        workflow = _get_compiled_workflow(llm_client)

        initial_state = {
            "question": question,
            "metadata": metadata,
            "conversation_history": conversation_history or [],
            "retry_count": 0,
            "max_retries": settings.limits.max_retries,
            "validation_errors": [],
        }
        final_state = workflow.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001 - last-resort guard around the whole pipeline
        logger.exception("Question answering failed unexpectedly")
        guidance = explain_error(exc, stage="workflow")
        return AnalysisResponse(
            status="error",
            question=question,
            error=guidance.reason,
            error_title=guidance.title,
            error_stage=guidance.stage,
            error_suggestions=guidance.suggestions,
            elapsed_seconds=time.monotonic() - started,
            llm_provider=selected_provider,
            llm_model=selected_model,
        )

    elapsed = time.monotonic() - started
    payload = final_state.get("final_response") or {}

    if payload.get("status") == "error":
        raw_error = payload.get("error") or "The query could not be answered."
        guidance = explain_error(raw_error, stage=payload.get("error_stage") or "query")
        return AnalysisResponse(
            status="error",
            question=question,
            sql=payload.get("sql"),
            error=guidance.reason,
            error_title=guidance.title,
            error_stage=guidance.stage,
            error_suggestions=guidance.suggestions,
            retry_count=payload.get("retry_count", 0),
            elapsed_seconds=elapsed,
            llm_provider=selected_provider,
            llm_model=selected_model,
        )

    insight = payload.get("insight") or {}
    preview_dataframe = payload.get("dataframe")
    download_dataframe: Optional[pd.DataFrame] = None
    download_row_count = 0
    download_truncated = False
    download_error = None
    download_sql = payload.get("download_sql")
    execution_sql = payload.get("execution_sql")

    if download_sql and download_sql == execution_sql:
        download_dataframe = preview_dataframe
        download_row_count = len(preview_dataframe) if preview_dataframe is not None else 0
    elif download_sql:
        export_result = execute_sql(
            download_sql,
            max_rows=settings.limits.download_max_rows,
            timeout_seconds=settings.limits.statement_timeout_seconds,
        )
        if export_result.success:
            download_dataframe = export_result.dataframe
            download_row_count = export_result.row_count
            download_truncated = export_result.truncated
        else:
            download_error = export_result.error or "The complete CSV could not be prepared."
    else:
        download_error = "No validated download query was available."

    response = AnalysisResponse(
        status="ok",
        question=question,
        sql=payload.get("sql"),
        sql_explanation=payload.get("sql_explanation"),
        dataframe=preview_dataframe,
        download_dataframe=download_dataframe,
        row_count=payload.get("row_count", 0),
        download_row_count=download_row_count,
        truncated=payload.get("truncated", False),
        download_truncated=download_truncated,
        download_error=download_error,
        insight_summary=insight.get("summary"),
        insight_findings=insight.get("key_findings", []),
        chart=payload.get("chart"),
        retry_count=payload.get("retry_count", 0),
        elapsed_seconds=elapsed,
        llm_provider=selected_provider,
        llm_model=selected_model,
        retrieval_mode=payload.get("retrieval_mode"),
    )

    if use_cache:
        _answer_cache[cache_key] = response
        _answer_cache.move_to_end(cache_key)
        if len(_answer_cache) > _ANSWER_CACHE_MAX_SIZE:
            _answer_cache.popitem(last=False)

    return response


def get_session_stats() -> dict[str, Any]:
    """Combined efficiency snapshot for the UI's sidebar panel."""
    usage = get_usage_stats()
    return {
        "llm_calls": usage.calls,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cached_questions": len(_answer_cache),
        "metadata_cache": get_metadata_cache_info(),
    }
