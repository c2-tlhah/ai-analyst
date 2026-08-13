"""Provider-aware UI copy for model operations.

Keep runtime promises here rather than scattering provider names through the
Streamlit layer. These descriptions are intentionally factual: they explain
where the selected model runs and which retry/queue behavior the backend
actually implements without guessing how long a request will take.
"""

from __future__ import annotations

from typing import Literal


Operation = Literal["data", "knowledge", "filesystem", "chart", "database"]

_PROVIDER_LABELS = {
    "azure_foundry": "Azure AI Foundry",
    "ollama": "Ollama (local)",
    "openrouter": "OpenRouter",
    "nvidia_nim": "NVIDIA NIM",
}

_OPERATION_ACTIONS: dict[Operation, str] = {
    "data": "is planning, validating, and answering the data question",
    "knowledge": "is answering from the schema and business-document context",
    "filesystem": "is working through the sandboxed filesystem tools",
    "chart": "is planning and validating the requested visualization",
    "database": "is enriching the discovered schema documentation",
}


def provider_label(provider: str | None) -> str:
    """Return one consistent user-facing provider label."""
    return _PROVIDER_LABELS.get(provider or "", provider or "Selected LLM")


def provider_runtime_note(provider: str | None, model: str | None) -> str:
    """Describe the selected runtime's real latency/retry behavior."""
    model_id = model or "selected model"
    if provider == "openrouter":
        route = "Free route" if model_id.casefold().endswith(":free") else "Hosted route"
        return (
            f"{route} through OpenRouter. Capacity is controlled by the upstream "
            "model provider; transient failures use bounded automatic retries."
        )
    if provider == "ollama":
        return (
            "Runs on this machine through Ollama. The first request can load the "
            "model into memory; generation speed depends on local CPU/GPU and model size."
        )
    if provider == "nvidia_nim":
        return (
            "Runs in NVIDIA NIM. Requests share the configured rolling-minute "
            "budget and are spaced or queued locally when that budget is busy."
        )
    if provider == "azure_foundry":
        reasoning = " Reasoning may add latency." if "kimi" in model_id.casefold() else ""
        return f"Runs in the configured Azure AI Foundry deployment.{reasoning}"
    return "Uses the selected model provider's configured runtime."


def operation_status(
    provider: str | None,
    model: str | None,
    operation: Operation,
) -> str:
    """Build status copy that always identifies the active provider and model."""
    action = _OPERATION_ACTIONS[operation]
    return (
        f"{provider_label(provider)} · {model or 'selected model'} {action}. "
        f"{provider_runtime_note(provider, model)}"
    )
