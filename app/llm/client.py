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
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import TypeVar
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

import requests
from pydantic import BaseModel, ValidationError

from app.config import AzureAIConfig, OllamaConfig, OpenRouterConfig, get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

AZURE_FOUNDRY_PROVIDER = "azure_foundry"
OLLAMA_PROVIDER = "ollama"
OPENROUTER_PROVIDER = "openrouter"
SUPPORTED_LLM_PROVIDERS = (AZURE_FOUNDRY_PROVIDER, OLLAMA_PROVIDER, OPENROUTER_PROVIDER)


class LLMError(RuntimeError):
    """Raised when the LLM cannot produce a usable response."""


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
        try:
            response = requests.post(
                f"{self._config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._config.request_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except requests.Timeout as exc:
            raise LLMError(
                f"OpenRouter request timed out after "
                f"{self._config.request_timeout_seconds}s."
            ) from exc
        except requests.RequestException as exc:
            detail = ""
            if getattr(exc, "response", None) is not None:
                try:
                    error_body = exc.response.json()
                    error_value = error_body.get("error", error_body)
                    detail = (
                        error_value.get("message", str(error_value))
                        if isinstance(error_value, dict)
                        else str(error_value)
                    )
                except (ValueError, AttributeError):
                    detail = exc.response.text[:500]
            raise LLMError(f"OpenRouter request failed: {detail or exc}") from exc
        except ValueError as exc:
            raise LLMError("OpenRouter returned an invalid JSON response.") from exc

        if not isinstance(body, dict):
            raise LLMError("OpenRouter returned an unexpected response body.")
        if body.get("error"):
            error_value = body["error"]
            message = (
                error_value.get("message", str(error_value))
                if isinstance(error_value, dict)
                else str(error_value)
            )
            raise LLMError(f"OpenRouter request failed: {message}")

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
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_LLM_PROVIDERS:
        raise LLMError(
            f"Unsupported LLM provider '{provider}'. Choose Azure AI Foundry, "
            "Ollama, or OpenRouter."
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
    else:
        selected_model = (model or settings.openrouter.model).strip()
        config = replace(settings.openrouter, model=selected_model)
        key = (provider_name, config.base_url, selected_model)
        factory = lambda: OpenRouterLLMClient(config)  # noqa: E731

    with _client_lock:
        client = _client_singletons.get(key)
        if client is None:
            client = factory()
            _client_singletons[key] = client
        return client
