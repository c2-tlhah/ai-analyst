"""Translate backend failures into clear, actionable user guidance.

The original exception/error text remains available as the reason, while this
module consistently supplies a short title and safe next steps for the UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorGuidance:
    title: str
    reason: str
    suggestions: tuple[str, ...]
    stage: str = "application"


def explain_error(error: object, *, stage: str = "application") -> ErrorGuidance:
    reason = str(error or "An unknown error occurred.").strip()
    lowered = reason.casefold()

    if stage == "input" or "please enter a question" in lowered:
        return ErrorGuidance(
            "A question is required",
            reason,
            (
                "Enter a business question that names a measure and a grouping.",
                "Example: Show monthly internet sales for the last year.",
            ),
            "input",
        )

    if stage == "validation":
        return ErrorGuidance(
            "A safe read-only query could not be generated",
            reason,
            (
                "Rephrase the question with the exact measure, grouping, and time range.",
                "Use names visible under Available data in the sidebar.",
                "Try a stronger model/provider if the selected model repeats invalid SQL.",
            ),
            "validation",
        )

    authentication_failure = any(
        token in lowered
        for token in ("401", "403", "authentication", "invalid api key", "invalid_api_key")
    ) or (stage == "provider" and "unauthorized" in lowered)
    if authentication_failure:
        return ErrorGuidance(
            "LLM authentication failed",
            reason,
            (
                "Verify the selected provider's API key in .env.",
                "Restart Streamlit after changing environment variables.",
                "Select another configured provider to continue immediately.",
            ),
            "provider",
        )

    if "not configured" in lowered or "set openrouter_api_key" in lowered or "set azure" in lowered:
        return ErrorGuidance(
            "The selected LLM is not configured",
            reason,
            (
                "Add the required endpoint/API key values to .env.",
                "Restart Streamlit so the new configuration is loaded.",
                "Select another provider that is marked Ready.",
            ),
            "configuration",
        )

    if "auto-start is disabled" in lowered:
        return ErrorGuidance(
            "Ollama is not running automatically",
            reason,
            (
                "Click Retry starting Ollama to start it for this session.",
                "Set OLLAMA_AUTO_START=true in .env and restart Streamlit.",
                "Or start ollama serve manually.",
            ),
            "service",
        )

    if "executable was not found" in lowered:
        return ErrorGuidance(
            "The Ollama executable was not found",
            reason,
            (
                "Install Ollama and restart Streamlit.",
                "Or set OLLAMA_EXECUTABLE to the full executable path in .env.",
                "Select a hosted provider until the local service is available.",
            ),
            "service",
        )

    if "remote endpoints" in lowered:
        return ErrorGuidance(
            "The remote Ollama endpoint is unavailable",
            reason,
            (
                "Start the remote Ollama service on its host.",
                "Verify OLLAMA_BASE_URL and network access.",
                "The app only auto-starts localhost Ollama instances.",
            ),
            "service",
        )

    if any(token in lowered for token in ("429", "rate limit", "quota", "too many requests")):
        return ErrorGuidance(
            "The LLM provider is temporarily rate-limited",
            reason,
            (
                "Wait briefly and retry the question.",
                "Select another model/provider in the sidebar.",
                "Check the provider account's quota or usage limits.",
            ),
            "provider",
        )

    if "timed out" in lowered or "timeout" in lowered or "time limit" in lowered:
        subject = "Database query" if stage in {"execution", "download"} else "LLM request"
        return ErrorGuidance(
            f"{subject} timed out",
            reason,
            (
                "Narrow the date range or ask for an aggregated result.",
                "Retry once in case the model or service was still loading.",
                "Increase the matching timeout setting in .env if the workload is expected.",
            ),
            stage,
        )

    if any(token in lowered for token in ("could not connect", "connection refused", "unavailable")):
        return ErrorGuidance(
            "The selected service is unavailable",
            reason,
            (
                "Check the service status and endpoint in the sidebar.",
                "For Ollama, retry startup and confirm the model is downloaded.",
                "Select another ready provider while the service is unavailable.",
            ),
            "service",
        )

    if "no downloaded models" in lowered or "no ollama model" in lowered:
        return ErrorGuidance(
            "No local model is available",
            reason,
            (
                "Run ollama pull <model>, for example ollama pull qwen2.5:7b.",
                "Refresh Ollama models in the sidebar after the download finishes.",
                "Select Azure AI Foundry or OpenRouter instead.",
            ),
            "configuration",
        )

    if "database not found" in lowered:
        return ErrorGuidance(
            "The analytics database was not found",
            reason,
            (
                "Run python scripts/build_database.py to create the sample database.",
                "Verify AI_ANALYST_DB_PATH in .env.",
            ),
            "database",
        )

    if any(
        token in lowered
        for token in (
            "sql failed to parse", "unauthorized table", "forbidden keyword",
            "only read-only select", "exactly one sql statement",
        )
    ):
        return ErrorGuidance(
            "A safe read-only query could not be generated",
            reason,
            (
                "Rephrase the question with the exact measure, grouping, and time range.",
                "Use names visible under Available data in the sidebar.",
                "Try a stronger model/provider if the selected model repeats invalid SQL.",
            ),
            "validation",
        )

    if stage == "download":
        return ErrorGuidance(
            "The complete CSV could not be prepared",
            reason,
            (
                "The visible analysis result is still available above.",
                "Narrow the query and try the download again.",
                "Check SQL_DOWNLOAD_MAX_ROWS and SQL_STATEMENT_TIMEOUT_SECONDS in .env.",
            ),
            "download",
        )

    if stage == "execution" or "database error" in lowered or "no such column" in lowered:
        return ErrorGuidance(
            "The validated query could not run",
            reason,
            (
                "Refresh the schema from the sidebar in case the database changed.",
                "Simplify the question or remove ambiguous field names.",
                "Review the displayed SQL and confirm the requested data exists.",
            ),
            "execution",
        )

    if "structured output" in lowered or "no assistant message" in lowered:
        return ErrorGuidance(
            "The model returned an unusable response",
            reason,
            (
                "Retry once; free and local models can occasionally return incomplete JSON.",
                "Select another model/provider if the problem repeats.",
                "Shorten or simplify the request.",
            ),
            "provider",
        )

    if stage == "configuration":
        return ErrorGuidance(
            "No ready LLM model is selected",
            reason,
            (
                "Choose a provider marked Ready in the sidebar.",
                "Configure its API credentials or download an Ollama model.",
                "Refresh the model list after changing the configuration.",
            ),
            "configuration",
        )

    if stage in {"provider", "service"}:
        return ErrorGuidance(
            "The selected AI service could not complete the request",
            reason,
            (
                "Retry once in case the service was temporarily unavailable.",
                "Check the provider/model status and configuration in the sidebar.",
                "Select another ready model/provider if the problem repeats.",
            ),
            stage,
        )

    if stage == "database":
        return ErrorGuidance(
            "The data source could not be accessed",
            reason,
            (
                "Verify AI_ANALYST_DB_PATH and that the database file is readable.",
                "Run python scripts/build_database.py if the sample database is missing.",
                "Refresh the schema after repairing the data source.",
            ),
            "database",
        )

    return ErrorGuidance(
        "The request could not be completed",
        reason,
        (
            "Retry the request once.",
            "Simplify the question and specify the measure, grouping, and date range.",
            "If it repeats, check logs/ai_analyst.log and try another provider.",
        ),
        stage,
    )
