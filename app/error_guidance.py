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

    dns_failure = any(
        token in lowered
        for token in (
            "nvidia_dns_resolution_failed",
            "nameresolutionerror",
            "failed to resolve",
            "getaddrinfo failed",
            "errno 11001",
            "dns lookup failed",
            "name resolution",
        )
    )
    if dns_failure:
        return ErrorGuidance(
            "DNS could not resolve the NVIDIA service",
            reason,
            (
                "Retry once; the app now retries transient NVIDIA DNS failures automatically.",
                "In PowerShell run: Resolve-DnsName integrate.api.nvidia.com",
                "If it times out, reconnect the network or VPN and run: ipconfig /flushdns",
                "If the router DNS remains unreliable, use a trusted DNS resolver or ask the network administrator to allow integrate.api.nvidia.com.",
                "Select Ollama, Azure AI Foundry, or OpenRouter while DNS is unavailable.",
            ),
            "network",
        )

    if stage == "input" or "please enter a question" in lowered:
        return ErrorGuidance(
            "A question is required",
            reason,
            (
                "Enter a business question that names a measure and a grouping.",
                "Example: Show a documented numeric measure by month for the last year.",
            ),
            "input",
        )

    if "no usable event-date field" in lowered or "no usable temporal" in lowered:
        return ErrorGuidance(
            "The requested time period is not supported by this data",
            reason,
            (
                "Name an exact date range and a date column visible in the active schema.",
                "Ask the knowledge base which temporal fields are available.",
                "If the source stores dates as text or numeric keys, rebuild metadata so the profiler can classify them.",
            ),
            "validation",
        )

    if "column resolution failed" in lowered:
        return ErrorGuidance(
            "The query referenced a column that is not available",
            reason,
            (
                "Use exact table and column names shown for the active database.",
                "Rebuild metadata if the database schema changed recently.",
                "Clarify which documented measure or category you mean.",
            ),
            "validation",
        )

    if "relationship" in lowered and "join" in lowered:
        return ErrorGuidance(
            "The requested tables do not have a verified join path",
            reason,
            (
                "Ask the knowledge base which relationships are declared or safely inferred.",
                "Name the intended key relationship explicitly if the database documentation is incomplete.",
                "Add a real foreign key or curate the database documentation before combining these tables.",
            ),
            "validation",
        )

    if "requires numeric data" in lowered or "not a discovered temporal field" in lowered:
        return ErrorGuidance(
            "The requested operation does not match the column data type",
            reason,
            (
                "Choose a numeric column for sums/averages or a temporal column for date grouping.",
                "Inspect the column profile and sample types in the database explorer.",
                "Clean mixed or incorrectly typed source values before retrying.",
            ),
            "validation",
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

    rate_limited = any(
        token in lowered
        for token in ("429", "rate limit", "rate-limit", "quota", "too many requests")
    )
    if rate_limited and "openrouter" in lowered:
        return ErrorGuidance(
            "The selected OpenRouter model is temporarily rate-limited",
            reason,
            (
                "Select another OpenRouter model in the sidebar; free-model providers have independent capacity.",
                "Wait briefly, then retry the original model.",
                "If every free model is limited, use Ollama/Azure or review OpenRouter account limits.",
            ),
            "provider",
        )

    if rate_limited and "nvidia" in lowered:
        return ErrorGuidance(
            "The NVIDIA request budget is temporarily exhausted",
            reason,
            (
                "Leave the request queued; the app spaces calls and resumes after NVIDIA's reset window.",
                "Check the NVIDIA request budget and request_budget events under Live agent logs.",
                "Avoid repeatedly clicking Refresh models or Test connection while a cooldown is active.",
                "Select another provider only when the request cannot wait for the current quota window.",
            ),
            "provider",
        )

    if rate_limited:
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
