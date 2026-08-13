"""Provider-neutral LLM clients with a JSON-structured-output contract.

Kimi K2.6 (like most Azure AI Foundry Model-Inference-API models) is reached
through the Azure AI Model Inference API. Rather than depend on
provider-specific structured-output/function-calling support (uneven across
the third-party models hosted on Foundry), we ask for JSON via the prompt,
parse it, and validate against a Pydantic schema -- with one bounded
"repair" retry that feeds the parse/validation error back to the model.
This keeps the contract portable across any Foundry chat-completion model.

Every node in the LangGraph workflow talks to :class:`LLMClient`, never to
a provider SDK directly, so tests can inject a deterministic fake and the
UI can switch providers without changing the analytics workflow.
"""

from __future__ import annotations

import json
import re
import socket
import threading
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import requests
from pydantic import BaseModel, ValidationError

from app.config import (
    AzureAIConfig,
    NvidiaNIMConfig,
    OllamaConfig,
    OpenRouterConfig,
    get_settings,
)
from app.logging_config import get_logger
from app.llm.rate_limiter import (
    RateLimitQueueTimeout,
    SlidingWindowRateLimiter,
)
from app.observability import emit_trace, trace_span

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

AZURE_FOUNDRY_PROVIDER = "azure_foundry"
OLLAMA_PROVIDER = "ollama"
OPENROUTER_PROVIDER = "openrouter"
NVIDIA_NIM_PROVIDER = "nvidia_nim"
SUPPORTED_LLM_PROVIDERS = (
    AZURE_FOUNDRY_PROVIDER,
    OLLAMA_PROVIDER,
    OPENROUTER_PROVIDER,
    NVIDIA_NIM_PROVIDER,
)

_nvidia_rate_limiters: dict[tuple[object, ...], SlidingWindowRateLimiter] = {}
_nvidia_rate_limiters_lock = threading.Lock()

NVIDIA_NIM_DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
NVIDIA_NIM_EXPLICIT_MODELS = (
    NVIDIA_NIM_DEFAULT_MODEL,
    "poolside/laguna-xs-2.1",
    "z-ai/glm-5.2",
    "minimaxai/minimax-m3",
)

NVIDIA_NIM_MODEL_CAPABILITIES = {
    NVIDIA_NIM_DEFAULT_MODEL: {
        "owner": "NVIDIA",
        "max_tokens": 16384,
        "reasoning_supported": True,
        "reasoning_controls": True,
        "tool_calling": True,
        "structured_output": False,
    },
    "poolside/laguna-xs-2.1": {
        "owner": "Poolside",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "reasoning_supported": True,
        "reasoning_controls": False,
        "tool_calling": True,
        "structured_output": False,
        "fixed_profile": True,
    },
    "z-ai/glm-5.2": {
        "owner": "Z.ai",
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 16384,
        "seed": 42,
        "reasoning_supported": True,
        "reasoning_controls": False,
        "tool_calling": True,
        "structured_output": True,
        "fixed_profile": True,
    },
    "minimaxai/minimax-m3": {
        "owner": "MiniMaxAI",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 8192,
        "reasoning_supported": True,
        "reasoning_controls": False,
        "tool_calling": True,
        "structured_output": True,
        "fixed_profile": True,
    },
}

# Curated models shown in the Streamlit selector. Metadata is refreshed from
# OpenRouter's public /models endpoint, while this stable ID list keeps the UI
# usable during a temporary metadata/network outage. Do not remove the earlier
# North Mini Code default when adding new choices.
OPENROUTER_FEATURED_MODELS = (
    "cohere/north-mini-code:free",
    "liquid/lfm-2.5-2.6b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "inclusionai/ling-3.0-tiny:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "nvidia/nemotron-3.5-content-safety:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-4-31b-it:free",
)


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a usable response."""


class LLMServiceError(LLMError):
    """Raised when a provider is unavailable before producing a response."""


class LLMRateLimitError(LLMServiceError):
    """Raised after bounded retries when a provider returns HTTP 429."""


@dataclass
class UsageStats:
    """Cumulative token usage for this process, across every LLM call.

    Real numbers straight from the API response (not estimated), so the UI
    can show an honest "how much are we actually spending" figure instead
    of a guess.
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_call_seconds: float = 0.0


_usage_stats = UsageStats()
_usage_lock = threading.Lock()


def get_usage_stats() -> UsageStats:
    with _usage_lock:
        return UsageStats(**_usage_stats.__dict__)


def reset_usage_stats() -> None:
    global _usage_stats
    with _usage_lock:
        _usage_stats = UsageStats()


def _record_usage(usage: object | None, elapsed_seconds: float) -> None:
    with _usage_lock:
        _usage_stats.calls += 1
        _usage_stats.total_call_seconds += elapsed_seconds
        if usage is not None:
            _usage_stats.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            _usage_stats.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            _usage_stats.total_tokens += getattr(usage, "total_tokens", 0) or 0


class LLMClient(ABC):
    """Minimal interface the rest of the backend depends on."""

    @abstractmethod
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        max_repair_attempts: int = 1,
    ) -> T:
        """Call the model and return a validated instance of ``schema``."""

    @abstractmethod
    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        """Call the model and return raw text (used for free-form insight copy)."""

    @property
    def cache_namespace(self) -> str:
        """Identify answers produced by this client inside the session cache."""
        return f"{type(self).__module__}.{type(self).__qualname__}:{id(self)}"

    @property
    def provider_name(self) -> str:
        return "custom"

    @property
    def model_name(self) -> str:
        return type(self).__name__

    @property
    def supports_tool_calling(self) -> bool:
        """Whether this concrete provider path implements native tool calls."""
        return False

    def complete_with_tools(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> dict:
        """Return a normalized assistant message containing optional tool calls."""
        raise LLMError(
            f"{self.provider_name} does not implement native function calling."
        )

    @property
    def supports_structured_output(self) -> bool:
        return False


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from a model response.

    Models frequently wrap JSON in markdown fences or add a sentence
    before/after it despite instructions; this pulls out the first
    top-level ``{...}`` block.
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        return text[brace_start : brace_end + 1]
    return text


def _validate_base_url(raw: str, *, provider_name: str) -> str:
    """Return a normalized HTTP(S) base URL or raise a user-facing error."""
    value = (raw or "").strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise LLMError(f"{provider_name} endpoint must be a valid HTTP(S) URL.")
    if parsed.query or parsed.fragment:
        raise LLMError(f"{provider_name} endpoint cannot contain a query string or fragment.")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _foundry_base_url(endpoint: str) -> str:
    """Normalize legacy Foundry URLs to the current OpenAI-compatible API root.

    Kept as a small public helper because deployments commonly surface either
    the resource root or the older ``/models`` inference endpoint.
    """
    base = _validate_base_url(endpoint, provider_name="Azure AI Foundry")
    parsed = urlsplit(base)
    path = parsed.path.rstrip("/")
    if "/api/projects/" in path:
        path = path.split("/api/projects/", 1)[0]
    if path.endswith("/models"):
        path = path[: -len("/models")]
    if path.endswith("/openai/v1"):
        final_path = path
    else:
        final_path = f"{path}/openai/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, f"{final_path}/", "", ""))


class AzureFoundryLLMClient(LLMClient):
    """Chat-completions client for an Azure AI Foundry serverless deployment."""

    def __init__(self, config: AzureAIConfig | None = None):
        self._config = config or get_settings().azure_ai
        if not self._config.is_configured:
            raise LLMError(
                "Azure AI Foundry is not configured. Set AZURE_FOUNDRY_ENDPOINT and "
                "AZURE_FOUNDRY_API_KEY (see .env.example)."
            )
        self._client = self._build_client()

    @property
    def cache_namespace(self) -> str:
        return (
            f"{AZURE_FOUNDRY_PROVIDER}:{self._config.endpoint}:"
            f"{self._config.model_deployment}"
        )

    @property
    def provider_name(self) -> str:
        return AZURE_FOUNDRY_PROVIDER

    @property
    def model_name(self) -> str:
        return self._config.model_deployment

    @property
    def supports_tool_calling(self) -> bool:
        # The configured Azure Kimi-K2.6 deployment was verified through the
        # Model Inference SDK's OpenAI-compatible tools surface. Other Azure
        # deployments remain opt-in by naming Kimi here rather than being
        # advertised as universally tool-capable.
        return "kimi-k2.6" in self._config.model_deployment.casefold()

    def _build_client(self):
        from azure.ai.inference import ChatCompletionsClient
        from azure.core.credentials import AzureKeyCredential

        kwargs = {}
        if self._config.api_version:
            kwargs["api_version"] = self._config.api_version
        return ChatCompletionsClient(
            endpoint=self._config.endpoint,
            credential=AzureKeyCredential(self._config.api_key),
            # Kimi K2.6's reasoning pass can take a while; give it real headroom
            # rather than the azure-core default (~5-10s connect / 100s read).
            connection_timeout=30,
            read_timeout=self._config.request_timeout_seconds,
            **kwargs,
        )

    def _call(self, system_prompt: str, user_prompt: str) -> str:
        from azure.ai.inference.models import SystemMessage, UserMessage
        from azure.core.exceptions import HttpResponseError

        started = time.monotonic()
        try:
            response = self._client.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt),
                ],
                model=self._config.model_deployment,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except HttpResponseError as exc:
            logger.error("Azure AI Foundry request failed: %s", exc)
            raise LLMError(f"LLM request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network/SDK errors of any shape
            logger.error("Azure AI Foundry request failed: %s", exc)
            raise LLMError(f"LLM request failed: {exc}") from exc

        _record_usage(getattr(response, "usage", None), time.monotonic() - started)
        return response.choices[0].message.content or ""

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._call(system_prompt, user_prompt).strip()

    def complete_with_tools(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> dict:
        """Call Azure Kimi with OpenAI-compatible function definitions."""
        if not self.supports_tool_calling:
            return super().complete_with_tools(
                messages=messages, tools=tools, tool_choice=tool_choice
            )
        from azure.core.exceptions import HttpResponseError

        started = time.monotonic()
        try:
            response = self._client.complete(
                messages=messages,
                model=self._config.model_deployment,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                tools=tools,
                tool_choice=tool_choice,
            )
        except HttpResponseError as exc:
            logger.error("Azure Kimi tool request failed: %s", exc)
            raise LLMServiceError(f"Azure Kimi tool request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - SDK/network errors are user-facing
            logger.error("Azure Kimi tool request failed: %s", exc)
            raise LLMServiceError(f"Azure Kimi tool request failed: {exc}") from exc

        _record_usage(getattr(response, "usage", None), time.monotonic() - started)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("Azure Kimi returned no assistant choice.")
        message = choices[0].message
        normalized = message.as_dict() if hasattr(message, "as_dict") else dict(message)
        normalized.setdefault("role", "assistant")
        return normalized

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        max_repair_attempts: int = 1,
    ) -> T:
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "Respond with ONLY a single valid JSON object -- no prose, no markdown "
            "fences, no explanation before or after. It must validate against this "
            f"JSON schema:\n{json.dumps(schema.model_json_schema())}"
        )

        raw = self._call(full_system_prompt, user_prompt)
        last_error: Exception | None = None

        for attempt in range(max_repair_attempts + 1):
            try:
                candidate = _extract_json(raw)
                data = json.loads(candidate)
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "LLM JSON output failed validation (attempt %d/%d): %s",
                    attempt + 1,
                    max_repair_attempts + 1,
                    exc,
                )
                if attempt >= max_repair_attempts:
                    break
                repair_prompt = (
                    "Your previous response was not valid JSON matching the schema. "
                    f"Error: {exc}\n\nPrevious response:\n{raw}\n\n"
                    "Return ONLY the corrected JSON object."
                )
                raw = self._call(full_system_prompt, repair_prompt)

        raise LLMError(f"LLM did not return valid structured output: {last_error}")


def _ollama_api_url(config: OllamaConfig, path: str) -> str:
    base = _validate_base_url(config.base_url, provider_name="Ollama")
    return f"{base}/api/{path.lstrip('/')}"


def _ollama_request(
    config: OllamaConfig,
    *,
    path: str,
    payload: dict | None = None,
    timeout_seconds: int,
) -> dict:
    """Call Ollama's local HTTP API using only the Python standard library."""
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        _ollama_api_url(config, path),
        data=data,
        method="POST" if payload is not None else "GET",
        headers={"Accept": "application/json", "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - configured endpoint
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - preserve the original HTTP failure
            detail = str(exc)
        raise LLMError(f"Ollama request failed ({exc.code}): {detail or exc.reason}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise LLMError(
            f"Could not connect to Ollama at {config.base_url}: {reason}. "
            "Make sure Ollama is installed and running."
        ) from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMError("Ollama returned an invalid JSON response.") from exc
    if not isinstance(parsed, dict):
        raise LLMError("Ollama returned an unexpected response.")
    if parsed.get("error"):
        raise LLMError(f"Ollama request failed: {parsed['error']}")
    return parsed


def list_ollama_models(config: OllamaConfig | None = None) -> list[str]:
    """Return the model names downloaded by the configured Ollama server."""
    config = config or get_settings().ollama
    response = _ollama_request(
        config,
        path="tags",
        timeout_seconds=config.discovery_timeout_seconds,
    )
    names: list[str] = []
    for item in response.get("models", []):
        if not isinstance(item, dict):
            continue
        name = item.get("model") or item.get("name")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return sorted(set(names), key=str.casefold)


def list_openrouter_model_details(
    config: OpenRouterConfig | None = None,
) -> dict[str, dict]:
    """Return current metadata for the app's curated OpenRouter models.

    OpenRouter's public model catalog is the authority for context length,
    modalities, pricing, and supported request parameters. The API key is
    optional for this metadata request but is included when configured.
    """
    config = config or get_settings().openrouter
    base_url = _validate_base_url(config.base_url, provider_name="OpenRouter")
    headers = {"Accept": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    try:
        response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=config.discovery_timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
    except requests.Timeout as exc:
        raise LLMError(
            f"OpenRouter model discovery timed out after "
            f"{config.discovery_timeout_seconds}s."
        ) from exc
    except requests.RequestException as exc:
        raise LLMError(f"OpenRouter model discovery failed: {exc}") from exc
    except ValueError as exc:
        raise LLMError("OpenRouter returned invalid model metadata.") from exc

    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list):
        raise LLMError("OpenRouter returned an unexpected model catalog.")

    wanted = set(OPENROUTER_FEATURED_MODELS)
    details: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict) or item.get("id") not in wanted:
            continue
        model_id = str(item["id"])
        architecture = item.get("architecture") or {}
        pricing = item.get("pricing") or {}
        supported_parameters = item.get("supported_parameters") or []
        details[model_id] = {
            "id": model_id,
            "name": str(item.get("name") or model_id),
            "description": str(item.get("description") or ""),
            "context_length": int(item.get("context_length") or 0),
            "input_modalities": list(architecture.get("input_modalities") or []),
            "output_modalities": list(architecture.get("output_modalities") or []),
            "supported_parameters": sorted(
                str(parameter) for parameter in supported_parameters
            ),
            "prompt_price": str(pricing.get("prompt") or ""),
            "completion_price": str(pricing.get("completion") or ""),
            "reasoning_supported": "reasoning" in supported_parameters,
            "structured_output_supported": bool(
                {"response_format", "structured_outputs"} & set(supported_parameters)
            ),
            "verified": True,
        }
    return details


def list_nvidia_nim_model_details(
    config: NvidiaNIMConfig | None = None,
) -> dict[str, dict]:
    """Discover only the NVIDIA-cloud models explicitly enabled by the user."""
    config = config or get_settings().nvidia_nim
    if not config.api_key:
        raise LLMError("NVIDIA NIM is not configured. Set NVIDIA_API_KEY in .env.")
    base_url = _validate_base_url(config.base_url, provider_name="NVIDIA NIM")
    body = _nvidia_request_json(
        config,
        method="get",
        path="/models",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Accept": "application/json",
        },
        timeout=config.discovery_timeout_seconds,
        operation="model_discovery",
        discovery=True,
    )

    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list):
        raise LLMError("NVIDIA NIM returned an unexpected model catalog.")

    details: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or model_id not in NVIDIA_NIM_EXPLICIT_MODELS:
            continue
        capabilities = NVIDIA_NIM_MODEL_CAPABILITIES[model_id]
        details[model_id] = {
            "id": model_id,
            "name": model_id,
            "owned_by": str(item.get("owned_by") or capabilities["owner"]),
            "created": item.get("created"),
            "max_tokens": capabilities["max_tokens"],
            "temperature": capabilities.get("temperature"),
            "top_p": capabilities.get("top_p"),
            "seed": capabilities.get("seed"),
            "reasoning_supported": capabilities["reasoning_supported"],
            "reasoning_controls_supported": capabilities["reasoning_controls"],
            "tool_calling_supported": capabilities["tool_calling"],
            "structured_output_supported": capabilities["structured_output"],
            "fixed_profile": bool(capabilities.get("fixed_profile")),
            "verified": True,
        }
    return dict(sorted(details.items(), key=lambda item: item[0].casefold()))


class OllamaLLMClient(LLMClient):
    """Chat client for a locally running Ollama instance."""

    def __init__(self, config: OllamaConfig | None = None):
        self._config = config or get_settings().ollama
        if not self._config.model:
            raise LLMError(
                "No Ollama model was selected. Download one with `ollama pull <model>` "
                "and select it in the sidebar."
            )
        # Validate eagerly so configuration problems appear before a long workflow starts.
        _validate_base_url(self._config.base_url, provider_name="Ollama")

    @property
    def cache_namespace(self) -> str:
        return f"{OLLAMA_PROVIDER}:{self._config.base_url}:{self._config.model}"

    @property
    def provider_name(self) -> str:
        return OLLAMA_PROVIDER

    @property
    def model_name(self) -> str:
        return self._config.model

    def _call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_schema: dict | None = None,
    ) -> str:
        payload: dict = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": self._config.temperature,
                "num_predict": self._config.max_tokens,
            },
        }
        if output_schema is not None:
            # Ollama accepts a JSON schema here and constrains generation to it.
            payload["format"] = output_schema

        started = time.monotonic()
        response = _ollama_request(
            self._config,
            path="chat",
            payload=payload,
            timeout_seconds=self._config.request_timeout_seconds,
        )
        prompt_tokens = response.get("prompt_eval_count", 0) or 0
        completion_tokens = response.get("eval_count", 0) or 0
        _record_usage(
            UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            time.monotonic() - started,
        )
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMError("Ollama returned no assistant message content.")
        return content

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        return self._call(system_prompt, user_prompt).strip()

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        max_repair_attempts: int = 1,
    ) -> T:
        json_schema = schema.model_json_schema()
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "Respond with ONLY a single valid JSON object -- no prose, no markdown "
            "fences, no explanation before or after. It must validate against this "
            f"JSON schema:\n{json.dumps(json_schema)}"
        )
        raw = self._call(full_system_prompt, user_prompt, output_schema=json_schema)
        last_error: Exception | None = None

        for attempt in range(max_repair_attempts + 1):
            try:
                data = json.loads(_extract_json(raw))
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "Ollama JSON output failed validation (attempt %d/%d): %s",
                    attempt + 1,
                    max_repair_attempts + 1,
                    exc,
                )
                if attempt >= max_repair_attempts:
                    break
                repair_prompt = (
                    "Your previous response was not valid JSON matching the schema. "
                    f"Error: {exc}\n\nPrevious response:\n{raw}\n\n"
                    "Return ONLY the corrected JSON object."
                )
                raw = self._call(
                    full_system_prompt,
                    repair_prompt,
                    output_schema=json_schema,
                )

        raise LLMError(f"LLM did not return valid structured output: {last_error}")


def _openrouter_error_detail(response: requests.Response) -> str:
    """Preserve useful nested upstream details from an OpenRouter error."""
    status_code = getattr(response, "status_code", None)
    pieces: list[str] = []
    try:
        body = response.json()
    except ValueError:
        body = None

    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        message = error.get("message")
        if message:
            pieces.append(str(message))
        error_code = error.get("code")
        if error_code and str(error_code) != str(status_code):
            pieces.append(f"code {error_code}")
        metadata = error.get("metadata")
        if isinstance(metadata, dict):
            provider = metadata.get("provider_name") or metadata.get("provider")
            if provider:
                pieces.append(f"provider: {provider}")
            raw = metadata.get("raw")
            if raw:
                upstream = str(raw)
                try:
                    raw_json = json.loads(upstream)
                    if isinstance(raw_json, dict):
                        nested_error = raw_json.get("error")
                        if isinstance(nested_error, dict):
                            upstream = str(
                                nested_error.get("message")
                                or nested_error.get("status")
                                or upstream
                            )
                except (json.JSONDecodeError, TypeError):
                    pass
                upstream = " ".join(upstream.split())[:600]
                if upstream and upstream not in pieces:
                    pieces.append(f"upstream: {upstream}")
    elif error:
        pieces.append(str(error))

    if not pieces:
        text = " ".join((getattr(response, "text", "") or "").split())[:600]
        if text:
            pieces.append(text)
    status = f"HTTP {status_code}" if status_code else "HTTP error"
    return f"{status}; " + "; ".join(pieces) if pieces else status


def _retry_delay(
    response: requests.Response | None,
    *,
    attempt: int,
    backoff_seconds: float,
) -> float:
    """Respect a short Retry-After value, otherwise use bounded backoff."""
    retry_after = None
    if response is not None:
        retry_after = response.headers.get("Retry-After")
    try:
        requested_delay = float(retry_after) if retry_after is not None else None
    except (TypeError, ValueError):
        requested_delay = None
    delay = (
        requested_delay
        if requested_delay is not None
        else max(0.0, backoff_seconds) * (2**attempt)
    )
    return min(max(0.0, delay), 5.0)


class OpenRouterLLMClient(LLMClient):
    """OpenRouter chat-completions client with reasoning and JSON Schema support."""

    def __init__(self, config: OpenRouterConfig | None = None):
        self._config = config or get_settings().openrouter
        if not self._config.is_configured:
            raise LLMError(
                "OpenRouter is not configured. Set OPENROUTER_API_KEY in .env."
            )
        _validate_base_url(self._config.base_url, provider_name="OpenRouter")

    @property
    def cache_namespace(self) -> str:
        return f"{OPENROUTER_PROVIDER}:{self._config.base_url}:{self._config.model}"

    @property
    def provider_name(self) -> str:
        return OPENROUTER_PROVIDER

    @property
    def model_name(self) -> str:
        return self._config.model

    def _call(
        self,
        messages: list[dict],
        *,
        output_schema: dict | None = None,
        schema_name: str = "structured_response",
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        if self._config.http_referer:
            headers["HTTP-Referer"] = self._config.http_referer
        if self._config.app_title:
            headers["X-OpenRouter-Title"] = self._config.app_title

        payload: dict = {
            "model": self._config.model,
            "messages": messages,
            "temperature": self._config.temperature,
            "max_tokens": self._config.max_tokens,
            "stream": False,
            "reasoning": {"enabled": self._config.reasoning_enabled},
        }
        if output_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": re.sub(r"[^a-zA-Z0-9_-]", "_", schema_name)[:64],
                    "strict": True,
                    "schema": output_schema,
                },
            }
            payload["plugins"] = [{"id": "response-healing"}]

        started = time.monotonic()
        body: dict | None = None
        max_attempts = max(1, self._config.max_retries + 1)
        for attempt in range(max_attempts):
            response: requests.Response | None = None
            try:
                response = requests.post(
                    f"{self._config.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self._config.request_timeout_seconds,
                )
                response.raise_for_status()
                parsed = response.json()
                body = parsed if isinstance(parsed, dict) else None
                break
            except requests.Timeout as exc:
                if attempt + 1 < max_attempts:
                    delay = _retry_delay(
                        response,
                        attempt=attempt,
                        backoff_seconds=self._config.retry_backoff_seconds,
                    )
                    logger.warning(
                        "OpenRouter timed out for %s; retrying in %.1fs (%d/%d)",
                        self._config.model,
                        delay,
                        attempt + 2,
                        max_attempts,
                    )
                    time.sleep(delay)
                    continue
                raise LLMServiceError(
                    f"OpenRouter timed out for {self._config.model} after "
                    f"{max_attempts} attempt(s) of "
                    f"{self._config.request_timeout_seconds}s."
                ) from exc
            except requests.RequestException as exc:
                failed_response = getattr(exc, "response", None)
                if failed_response is None:
                    failed_response = response
                status_code = getattr(failed_response, "status_code", None)
                retryable = status_code in {408, 409, 429, 500, 502, 503, 504} or (
                    status_code is None
                )
                detail = (
                    _openrouter_error_detail(failed_response)
                    if failed_response is not None
                    else str(exc)
                )
                if retryable and attempt + 1 < max_attempts:
                    delay = _retry_delay(
                        failed_response,
                        attempt=attempt,
                        backoff_seconds=self._config.retry_backoff_seconds,
                    )
                    logger.warning(
                        "OpenRouter transient failure for %s (%s); retrying in %.1fs (%d/%d)",
                        self._config.model,
                        detail,
                        delay,
                        attempt + 2,
                        max_attempts,
                    )
                    time.sleep(delay)
                    continue
                message = (
                    f"OpenRouter rate limit for {self._config.model} after "
                    f"{max_attempts} attempt(s): {detail}"
                    if status_code == 429
                    else f"OpenRouter request failed for {self._config.model}: {detail}"
                )
                error_type = LLMRateLimitError if status_code == 429 else LLMServiceError
                raise error_type(message) from exc
            except ValueError as exc:
                raise LLMServiceError(
                    "OpenRouter returned an invalid JSON response."
                ) from exc

        if not isinstance(body, dict):
            raise LLMServiceError("OpenRouter returned an unexpected response body.")
        if body.get("error"):
            error_value = body["error"]
            message = (
                error_value.get("message", str(error_value))
                if isinstance(error_value, dict)
                else str(error_value)
            )
            code = error_value.get("code") if isinstance(error_value, dict) else None
            error_type = LLMRateLimitError if code == 429 else LLMServiceError
            raise error_type(
                f"OpenRouter request failed for {self._config.model}: {message}"
            )

        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        _record_usage(
            UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=usage.get("total_tokens", prompt_tokens + completion_tokens) or 0,
            ),
            time.monotonic() - started,
        )

        choices = body.get("choices") or []
        message = choices[0].get("message") if choices else None
        if not isinstance(message, dict):
            raise LLMError("OpenRouter returned no assistant message.")
        return message

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        message = self._call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError("OpenRouter returned no assistant message content.")
        return content.strip()

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        max_repair_attempts: int = 1,
    ) -> T:
        json_schema = schema.model_json_schema()
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "Return only a JSON object matching the supplied response schema. "
            "Use the exact property names shown; include every required property, "
            "and use schema defaults instead of null for enum fields.\n"
            f"Required JSON schema:\n{json.dumps(json_schema)}"
        )
        messages: list[dict] = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        message = self._call(
            messages,
            output_schema=json_schema,
            schema_name=schema.__name__,
        )
        last_error: Exception | None = None

        for attempt in range(max_repair_attempts + 1):
            raw = message.get("content") or ""
            try:
                data = json.loads(_extract_json(raw))
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "OpenRouter JSON output failed validation (attempt %d/%d): %s",
                    attempt + 1,
                    max_repair_attempts + 1,
                    exc,
                )
                if attempt >= max_repair_attempts:
                    break

                # Preserve reasoning_details unmodified in the assistant turn, as
                # required by OpenRouter for reasoning continuation.
                assistant_message = {"role": "assistant", "content": raw}
                if "reasoning_details" in message:
                    assistant_message["reasoning_details"] = message["reasoning_details"]
                messages.extend(
                    [
                        assistant_message,
                        {
                            "role": "user",
                            "content": (
                                "The previous JSON did not match the schema. "
                                f"Validation error: {exc}. Return only corrected JSON."
                            ),
                        },
                    ]
                )
                message = self._call(
                    messages,
                    output_schema=json_schema,
                    schema_name=schema.__name__,
                )

        raise LLMError(f"LLM did not return valid structured output: {last_error}")


def _nvidia_error_detail(response: requests.Response) -> str:
    status_code = getattr(response, "status_code", None)
    detail = ""
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        error = body.get("error") or body.get("detail")
        if isinstance(error, dict):
            detail = str(
                error.get("message")
                or error.get("detail")
                or error.get("type")
                or error
            )
            code = error.get("code")
            if code and str(code) not in detail:
                detail = f"{detail} (code {code})"
        elif error:
            detail = str(error)
    if not detail:
        detail = (getattr(response, "text", "") or "").strip()[:600]
    status = f"HTTP {status_code}" if status_code else "HTTP error"
    return f"{status}; {' '.join(detail.split())[:600]}" if detail else status


def _is_dns_resolution_error(exc: BaseException) -> bool:
    """Recognize requests/urllib3/socket DNS failures without importing internals."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}".casefold()
        if any(
            marker in text
            for marker in (
                "nameresolutionerror",
                "failed to resolve",
                "getaddrinfo failed",
                "errno 11001",
                "name or service not known",
                "nodename nor servname",
                "temporary failure in name resolution",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _nvidia_dns_error(config: NvidiaNIMConfig, attempts: int) -> str:
    host = urlsplit(config.base_url).hostname or "integrate.api.nvidia.com"
    return (
        f"NVIDIA NIM DNS lookup failed for {host} after {attempts} attempt(s) "
        "(NVIDIA_DNS_RESOLUTION_FAILED). This computer's DNS service could not "
        "resolve the NVIDIA endpoint, so the model and API key were not contacted."
    )


def _nvidia_rate_limiter(config: NvidiaNIMConfig) -> SlidingWindowRateLimiter:
    """Return the shared quota gate for one NVIDIA endpoint/API account."""
    credential_fingerprint = hashlib.sha256(
        config.api_key.encode("utf-8")
    ).hexdigest()[:16]
    key = (
        config.base_url.casefold(),
        credential_fingerprint,
        config.requests_per_minute,
        config.rate_limit_window_seconds,
        config.min_request_interval_seconds,
        config.rate_limit_max_wait_seconds,
    )
    with _nvidia_rate_limiters_lock:
        limiter = _nvidia_rate_limiters.get(key)
        if limiter is None:
            limiter = SlidingWindowRateLimiter(
                limit=config.requests_per_minute,
                window_seconds=config.rate_limit_window_seconds,
                min_interval_seconds=config.min_request_interval_seconds,
                max_wait_seconds=config.rate_limit_max_wait_seconds,
            )
            _nvidia_rate_limiters[key] = limiter
        return limiter


def get_nvidia_rate_limit_status(
    config: NvidiaNIMConfig | None = None,
) -> dict[str, float | int]:
    """Safe process-local NVIDIA request-budget telemetry for the UI."""
    config = config or get_settings().nvidia_nim
    if not config.api_key:
        return {
            "limit": max(1, config.requests_per_minute),
            "requests_in_window": 0,
            "remaining_requests": max(1, config.requests_per_minute),
            "window_seconds": config.rate_limit_window_seconds,
            "next_request_in_seconds": 0.0,
            "cooldown_seconds": 0.0,
        }
    return _nvidia_rate_limiter(config).status()


def _retry_after_seconds(
    response: requests.Response | None,
    *,
    default: float,
) -> float:
    value = response.headers.get("Retry-After") if response is not None else None
    try:
        return max(0.0, float(value)) if value is not None else max(0.0, default)
    except (TypeError, ValueError):
        return max(0.0, default)


def _safe_rate_headers(response: requests.Response | None) -> dict[str, str]:
    if response is None:
        return {}
    wanted = {
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
    }
    return {
        key.casefold(): str(value)[:100]
        for key, value in response.headers.items()
        if key.casefold() in wanted
    }


def _nvidia_request_json(
    config: NvidiaNIMConfig,
    *,
    method: str,
    path: str,
    headers: dict[str, str],
    timeout: int,
    operation: str,
    payload: dict | None = None,
    discovery: bool = False,
) -> dict:
    """Execute a bounded, traced NVIDIA request with DNS-aware retry errors."""
    url = f"{config.base_url}{path}"
    host = urlsplit(config.base_url).hostname or "integrate.api.nvidia.com"
    max_attempts = max(1, config.max_retries + 1)
    metadata = {
        "provider": NVIDIA_NIM_PROVIDER,
        "model": None if discovery else config.model,
        "host": host,
        "path": path,
        "max_attempts": max_attempts,
        "timeout_seconds": timeout,
    }
    with trace_span(operation, category="llm", metadata=metadata):
        limiter = _nvidia_rate_limiter(config)
        for attempt in range(max_attempts):
            response: requests.Response | None = None
            try:
                budget_before = limiter.status()
                if float(budget_before.get("next_request_in_seconds", 0)) >= 0.05:
                    emit_trace(
                        "request_budget",
                        category="provider",
                        status="queued",
                        message=(
                            "Waiting for the shared NVIDIA request budget before "
                            "the next HTTP attempt."
                        ),
                        metadata={
                            "provider": NVIDIA_NIM_PROVIDER,
                            **budget_before,
                        },
                    )
                try:
                    permit = limiter.acquire()
                except RateLimitQueueTimeout as exc:
                    emit_trace(
                        "request_budget",
                        category="provider",
                        status="failed",
                        message=str(exc),
                        metadata={
                            "provider": NVIDIA_NIM_PROVIDER,
                            **limiter.status(),
                        },
                    )
                    raise LLMRateLimitError(
                        "NVIDIA NIM local rate-limit request queue timed out before an HTTP "
                        f"attempt: {exc}"
                    ) from exc
                emit_trace(
                    "request_budget",
                    category="provider",
                    status="completed",
                    message=(
                        f"Queued for {permit.waited_seconds:.1f}s to stay within "
                        f"{permit.limit} requests/minute."
                        if permit.waited_seconds >= 0.05
                        else "NVIDIA request permit granted."
                    ),
                    metadata={
                        "provider": NVIDIA_NIM_PROVIDER,
                        "waited_seconds": round(permit.waited_seconds, 3),
                        "requests_in_window": permit.requests_in_window,
                        "remaining_requests": permit.remaining_requests,
                        "limit": permit.limit,
                    },
                )
                kwargs: dict = {"headers": headers, "timeout": timeout}
                if payload is not None:
                    kwargs["json"] = payload
                request_fn = requests.get if method.casefold() == "get" else requests.post
                response = request_fn(url, **kwargs)
                response.raise_for_status()
                parsed = response.json()
                if not isinstance(parsed, dict):
                    raise ValueError("response was not a JSON object")
                emit_trace(
                    "http_attempt",
                    category="provider",
                    status="completed",
                    metadata={
                        "provider": NVIDIA_NIM_PROVIDER,
                        "operation": operation,
                        "attempt": attempt + 1,
                        "http_status": response.status_code,
                        "rate_headers": _safe_rate_headers(response),
                        **limiter.status(),
                    },
                )
                return parsed
            except requests.Timeout as exc:
                if attempt + 1 < max_attempts:
                    delay = _retry_delay(
                        response,
                        attempt=attempt,
                        backoff_seconds=config.retry_backoff_seconds,
                    )
                    emit_trace(
                        "http_attempt",
                        category="provider",
                        status="retrying",
                        message="NVIDIA request timed out.",
                        metadata={"attempt": attempt + 1, "retry_in_seconds": delay},
                    )
                    time.sleep(delay)
                    continue
                message = (
                    f"NVIDIA NIM model discovery timed out after {max_attempts} attempt(s)."
                    if discovery
                    else f"NVIDIA NIM timed out for {config.model} after {max_attempts} attempt(s)."
                )
                error_type = LLMError if discovery else LLMServiceError
                raise error_type(message) from exc
            except requests.RequestException as exc:
                failed_response = getattr(exc, "response", None) or response
                status_code = getattr(failed_response, "status_code", None)
                dns_failure = _is_dns_resolution_error(exc)
                detail = (
                    _nvidia_dns_error(config, attempt + 1)
                    if dns_failure
                    else (
                        _nvidia_error_detail(failed_response)
                        if failed_response is not None
                        else str(exc)
                    )
                )
                retryable = dns_failure or status_code in {
                    408, 409, 429, 500, 502, 503, 504
                } or status_code is None
                rate_cooldown = 0.0
                if status_code == 429:
                    rate_cooldown = _retry_after_seconds(
                        failed_response,
                        default=config.rate_limit_429_cooldown_seconds,
                    )
                    limiter.penalize(rate_cooldown)
                if retryable and attempt + 1 < max_attempts:
                    if status_code == 429:
                        delay = rate_cooldown
                    else:
                        delay = _retry_delay(
                            failed_response,
                            attempt=attempt,
                            backoff_seconds=config.retry_backoff_seconds,
                        )
                    logger.warning(
                        "NVIDIA NIM transient %s failure for %s; retrying in %.1fs (%d/%d)",
                        "DNS" if dns_failure else "HTTP",
                        config.model,
                        delay,
                        attempt + 2,
                        max_attempts,
                    )
                    emit_trace(
                        "http_attempt",
                        category="provider",
                        status="retrying",
                        message=detail,
                        metadata={
                            "attempt": attempt + 1,
                            "retry_in_seconds": delay,
                            "http_status": status_code,
                            "dns_failure": dns_failure,
                            "shared_cooldown": status_code == 429,
                            "rate_headers": _safe_rate_headers(failed_response),
                        },
                    )
                    # A 429 cooldown is enforced by the shared limiter on the
                    # next loop and across every concurrent Streamlit session.
                    if status_code != 429:
                        time.sleep(delay)
                    continue
                if dns_failure:
                    detail = _nvidia_dns_error(config, attempt + 1)
                prefix = (
                    "NVIDIA NIM model discovery failed"
                    if discovery
                    else f"NVIDIA NIM request failed for {config.model}"
                )
                error_type = (
                    LLMError
                    if discovery
                    else LLMRateLimitError
                    if status_code == 429
                    else LLMServiceError
                )
                raise error_type(f"{prefix}: {detail}") from exc
            except ValueError as exc:
                message = (
                    "NVIDIA NIM returned invalid model metadata."
                    if discovery
                    else "NVIDIA NIM returned an invalid JSON response."
                )
                error_type = LLMError if discovery else LLMServiceError
                raise error_type(message) from exc

    raise LLMServiceError("NVIDIA NIM request stopped without a response.")


def check_nvidia_nim_health(
    config: NvidiaNIMConfig | None = None,
) -> dict[str, object]:
    """Verify local DNS, NVIDIA authentication/catalog access, and selected model."""
    config = config or get_settings().nvidia_nim
    if not config.is_configured:
        raise LLMError("NVIDIA NIM is not configured. Set NVIDIA_API_KEY in .env.")
    host = urlsplit(_validate_base_url(config.base_url, provider_name="NVIDIA NIM")).hostname
    started = time.monotonic()
    try:
        addresses = sorted(
            {
                item[4][0]
                for item in socket.getaddrinfo(
                    host or "integrate.api.nvidia.com",
                    443,
                    type=socket.SOCK_STREAM,
                )
            }
        )
    except OSError as exc:
        raise LLMServiceError(_nvidia_dns_error(config, 1)) from exc
    details = list_nvidia_nim_model_details(config)
    return {
        "ok": True,
        "host": host,
        "addresses": addresses,
        "model": config.model,
        "model_available": config.model in details,
        "catalog_model_count": len(details),
        "elapsed_seconds": time.monotonic() - started,
    }


class NvidiaNIMLLMClient(LLMClient):
    """NVIDIA API Catalog/NIM client using its OpenAI-compatible HTTP API."""

    def __init__(self, config: NvidiaNIMConfig | None = None):
        self._config = config or get_settings().nvidia_nim
        if not self._config.is_configured:
            raise LLMError(
                "NVIDIA NIM is not configured. Set NVIDIA_API_KEY in .env."
            )
        _validate_base_url(self._config.base_url, provider_name="NVIDIA NIM")

    @property
    def cache_namespace(self) -> str:
        return f"{NVIDIA_NIM_PROVIDER}:{self._config.base_url}:{self._config.model}"

    @property
    def provider_name(self) -> str:
        return NVIDIA_NIM_PROVIDER

    @property
    def model_name(self) -> str:
        return self._config.model

    @property
    def supports_tool_calling(self) -> bool:
        return bool(
            NVIDIA_NIM_MODEL_CAPABILITIES.get(self._config.model, {}).get(
                "tool_calling"
            )
        )

    @property
    def supports_structured_output(self) -> bool:
        return bool(
            NVIDIA_NIM_MODEL_CAPABILITIES.get(self._config.model, {}).get(
                "structured_output"
            )
        )

    def _call(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        capabilities = NVIDIA_NIM_MODEL_CAPABILITIES.get(self._config.model, {})
        fixed_profile = bool(capabilities.get("fixed_profile"))
        model_max_tokens = int(capabilities.get("max_tokens") or self._config.max_tokens)
        payload: dict = {
            "model": self._config.model,
            "messages": messages,
            "temperature": (
                capabilities.get("temperature", self._config.temperature)
                if fixed_profile
                else self._config.temperature
            ),
            "top_p": (
                capabilities.get("top_p", self._config.top_p)
                if fixed_profile
                else self._config.top_p
            ),
            "max_tokens": (
                model_max_tokens
                if fixed_profile
                else min(self._config.max_tokens, model_max_tokens)
            ),
            # The analytics workflow must validate a complete JSON object before
            # any SQL is used, so backend calls intentionally do not stream.
            "stream": False,
        }
        if fixed_profile and capabilities.get("seed") is not None:
            payload["seed"] = capabilities["seed"]
        if self._config.reasoning_enabled and capabilities.get(
            "reasoning_controls", True
        ):
            payload["chat_template_kwargs"] = {"enable_thinking": True}
            payload["reasoning_budget"] = self._config.reasoning_budget
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
            if self._config.model == NVIDIA_NIM_DEFAULT_MODEL:
                payload.setdefault("chat_template_kwargs", {})[
                    "force_nonempty_content"
                ] = True

        started = time.monotonic()
        body = _nvidia_request_json(
            self._config,
            method="post",
            path="/chat/completions",
            headers=headers,
            payload=payload,
            timeout=self._config.request_timeout_seconds,
            operation="chat_completion",
        )
        if body.get("error"):
            error = body["error"]
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            code = error.get("code") if isinstance(error, dict) else None
            error_type = LLMRateLimitError if code == 429 else LLMServiceError
            raise error_type(
                f"NVIDIA NIM request failed for {self._config.model}: {message}"
            )

        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        _record_usage(
            UsageStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=usage.get(
                    "total_tokens", prompt_tokens + completion_tokens
                )
                or 0,
            ),
            time.monotonic() - started,
        )

        choices = body.get("choices") or []
        message = choices[0].get("message") if choices else None
        if not isinstance(message, dict):
            raise LLMError("NVIDIA NIM returned no assistant message.")
        return message

    def complete_text(self, *, system_prompt: str, user_prompt: str) -> str:
        message = self._call(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            if message.get("reasoning_content"):
                raise LLMError(
                    "NVIDIA NIM used the available output budget for reasoning but "
                    "returned no final answer. Increase NVIDIA_NIM_MAX_TOKENS or "
                    "reduce NVIDIA_NIM_REASONING_BUDGET."
                )
            raise LLMError("NVIDIA NIM returned no assistant message content.")
        return content.strip()

    def complete_with_tools(
        self,
        *,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
    ) -> dict:
        if not self.supports_tool_calling:
            return super().complete_with_tools(
                messages=messages, tools=tools, tool_choice=tool_choice
            )
        return self._call(messages, tools=tools, tool_choice=tool_choice)

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        max_repair_attempts: int = 1,
    ) -> T:
        json_schema = schema.model_json_schema()
        full_system_prompt = (
            f"{system_prompt}\n\n"
            "Return only one valid JSON object with no markdown or surrounding prose. "
            "Use the exact property names, include every required property, and use "
            "schema defaults instead of null for enum fields.\n"
            f"Required JSON schema:\n{json.dumps(json_schema)}"
        )
        response_format = None
        if self.supports_structured_output:
            schema_name = re.sub(r"[^A-Za-z0-9_-]", "_", schema.__name__)[:64]
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or "structured_result",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        def call_for_json(prompt: str) -> str:
            if not response_format:
                return self.complete_text(
                    system_prompt=full_system_prompt, user_prompt=prompt
                )
            message = self._call(
                [
                    {"role": "system", "content": full_system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format=response_format,
            )
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                raise LLMError(
                    "NVIDIA NIM returned no content for a structured-output request."
                )
            return content.strip()

        raw = call_for_json(user_prompt)
        last_error: Exception | None = None
        for attempt in range(max_repair_attempts + 1):
            try:
                return schema.model_validate(json.loads(_extract_json(raw)))
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "NVIDIA NIM JSON output failed validation (attempt %d/%d): %s",
                    attempt + 1,
                    max_repair_attempts + 1,
                    exc,
                )
                if attempt >= max_repair_attempts:
                    break
                raw = call_for_json(
                    (
                        "The previous response did not match the required JSON schema. "
                        f"Validation error: {exc}. Previous response: {raw}. "
                        "Return only the corrected JSON object."
                    )
                )
        raise LLMError(f"LLM did not return valid structured output: {last_error}")


_client_singletons: dict[tuple[str, str, str], LLMClient] = {}
_client_lock = threading.Lock()


def _normalize_provider(provider: str | None) -> str:
    value = (provider or get_settings().default_llm_provider).strip().lower().replace("-", "_")
    aliases = {
        "azure": AZURE_FOUNDRY_PROVIDER,
        "foundry": AZURE_FOUNDRY_PROVIDER,
        "azure_ai_foundry": AZURE_FOUNDRY_PROVIDER,
        "local": OLLAMA_PROVIDER,
        "open_router": OPENROUTER_PROVIDER,
        "nvidia": NVIDIA_NIM_PROVIDER,
        "nim": NVIDIA_NIM_PROVIDER,
        "nvidia_cloud": NVIDIA_NIM_PROVIDER,
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_LLM_PROVIDERS:
        raise LLMError(
            f"Unsupported LLM provider '{provider}'. Choose Azure AI Foundry, "
            "Ollama, OpenRouter, or NVIDIA NIM."
        )
    return value


def get_llm_client(provider: str | None = None, model: str | None = None) -> LLMClient:
    """Return a reusable client for the selected provider and model.

    Calling this without arguments preserves the original behavior: Azure AI
    Foundry and the configured Kimi deployment remain the default.
    """
    settings = get_settings()
    provider_name = _normalize_provider(provider)

    if provider_name == AZURE_FOUNDRY_PROVIDER:
        selected_model = (model or settings.azure_ai.model_deployment).strip()
        config = replace(settings.azure_ai, model_deployment=selected_model)
        key = (provider_name, config.endpoint, selected_model)
        factory = lambda: AzureFoundryLLMClient(config)  # noqa: E731
    elif provider_name == OLLAMA_PROVIDER:
        selected_model = (model or settings.ollama.model).strip()
        if not selected_model:
            downloaded = list_ollama_models(settings.ollama)
            if not downloaded:
                raise LLMError(
                    "Ollama is running, but it has no downloaded models. "
                    "Run `ollama pull <model>` first."
                )
            selected_model = downloaded[0]
        config = replace(settings.ollama, model=selected_model)
        key = (provider_name, config.base_url, selected_model)
        factory = lambda: OllamaLLMClient(config)  # noqa: E731
    elif provider_name == OPENROUTER_PROVIDER:
        selected_model = (model or settings.openrouter.model).strip()
        config = replace(settings.openrouter, model=selected_model)
        key = (provider_name, config.base_url, selected_model)
        factory = lambda: OpenRouterLLMClient(config)  # noqa: E731
    else:
        selected_model = (model or settings.nvidia_nim.model).strip()
        config = replace(settings.nvidia_nim, model=selected_model)
        key = (provider_name, config.base_url, selected_model)
        factory = lambda: NvidiaNIMLLMClient(config)  # noqa: E731

    with _client_lock:
        client = _client_singletons.get(key)
        if client is None:
            client = factory()
            _client_singletons[key] = client
        return client
