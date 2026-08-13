import json
from types import SimpleNamespace

import pytest
import requests
from pydantic import BaseModel

from app.config import AzureAIConfig, NvidiaNIMConfig, OllamaConfig, OpenRouterConfig
from app.llm.client import (
    AzureFoundryLLMClient,
    LLMError,
    LLMRateLimitError,
    LLMServiceError,
    NvidiaNIMLLMClient,
    OPENROUTER_FEATURED_MODELS,
    OllamaLLMClient,
    OpenRouterLLMClient,
    _foundry_base_url,
    list_openrouter_model_details,
    list_nvidia_nim_model_details,
    list_ollama_models,
)


class _Response:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _StructuredAnswer(BaseModel):
    answer: str


class _RequestsResponse:
    def __init__(self, payload, *, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} response",
                response=self,
            )
        return None

    def json(self):
        return self._payload


def test_foundry_endpoint_is_normalized_to_openai_v1():
    assert _foundry_base_url("https://example.services.ai.azure.com") == (
        "https://example.services.ai.azure.com/openai/v1/"
    )


def test_legacy_models_endpoint_is_migrated_non_breakingly():
    assert _foundry_base_url("https://example.services.ai.azure.com/models") == (
        "https://example.services.ai.azure.com/openai/v1/"
    )


def test_existing_v1_endpoint_is_preserved():
    assert _foundry_base_url(
        "https://example.services.ai.azure.com/openai/v1/"
    ) == "https://example.services.ai.azure.com/openai/v1/"


@pytest.mark.parametrize("endpoint", ["not-a-url", "", "ftp://example.test/models"])
def test_invalid_foundry_endpoint_is_rejected(endpoint):
    with pytest.raises(LLMError):
        _foundry_base_url(endpoint)


def test_ollama_lists_downloaded_models(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["timeout"] = timeout
        return _Response(
            {
                "models": [
                    {"name": "qwen2.5:7b", "model": "qwen2.5:7b"},
                    {"name": "llama3.2:latest"},
                ]
            }
        )

    monkeypatch.setattr("app.llm.client.urlopen", fake_urlopen)
    config = OllamaConfig(base_url="http://localhost:11434", discovery_timeout_seconds=4)

    assert list_ollama_models(config) == ["llama3.2:latest", "qwen2.5:7b"]
    assert captured == {
        "url": "http://localhost:11434/api/tags",
        "method": "GET",
        "timeout": 4,
    }


def test_ollama_chat_uses_selected_model_and_structured_format(monkeypatch):
    payloads = []

    def fake_urlopen(request, timeout):
        assert timeout == 12
        payloads.append(json.loads(request.data.decode("utf-8")))
        return _Response(
            {
                "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                "prompt_eval_count": 11,
                "eval_count": 3,
            }
        )

    monkeypatch.setattr("app.llm.client.urlopen", fake_urlopen)
    client = OllamaLLMClient(
        OllamaConfig(
            base_url="http://localhost:11434",
            model="qwen2.5:7b",
            request_timeout_seconds=12,
        )
    )

    result = client.complete_json(
        system_prompt="Return an answer.",
        user_prompt="Test",
        schema=_StructuredAnswer,
    )

    assert result.answer == "ok"
    assert payloads[0]["model"] == "qwen2.5:7b"
    assert payloads[0]["stream"] is False
    assert payloads[0]["format"]["properties"]["answer"]["type"] == "string"


def test_nvidia_nim_discovers_nvidia_owned_models(monkeypatch):
    captured = {}

    def fake_get(url, *, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return _RequestsResponse(
            {
                "data": [
                    {
                        "id": "nvidia/nemotron-3-ultra-550b-a55b",
                        "owned_by": "nvidia",
                        "created": 123,
                    },
                    {
                        "id": "poolside/laguna-xs-2.1",
                        "owned_by": "poolside",
                    },
                    {"id": "z-ai/glm-5.2", "owned_by": "z-ai"},
                    {"id": "minimaxai/minimax-m3", "owned_by": "minimaxai"},
                    {"id": "nvidia/unrequested-model", "owned_by": "nvidia"},
                    {"id": "meta/llama-test", "owned_by": "meta"},
                ]
            }
        )

    monkeypatch.setattr("app.llm.client.requests.get", fake_get)
    details = list_nvidia_nim_model_details(
        NvidiaNIMConfig(api_key="nvapi-test", discovery_timeout_seconds=6)
    )

    assert list(details) == [
        "minimaxai/minimax-m3",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "poolside/laguna-xs-2.1",
        "z-ai/glm-5.2",
    ]
    assert details["nvidia/nemotron-3-ultra-550b-a55b"]["verified"] is True
    assert captured == {
        "url": "https://integrate.api.nvidia.com/v1/models",
        "headers": {
            "Authorization": "Bearer nvapi-test",
            "Accept": "application/json",
        },
        "timeout": 6,
    }


def test_nvidia_nim_sends_reasoning_parameters_and_non_streaming_request(monkeypatch):
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _RequestsResponse(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": '{"answer":"ok"}'}}
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
            }
        )

    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    client = NvidiaNIMLLMClient(
        NvidiaNIMConfig(
            api_key="nvapi-test",
            model="nvidia/nemotron-3-ultra-550b-a55b",
            temperature=1.0,
            top_p=0.95,
            max_tokens=16384,
            reasoning_enabled=True,
            reasoning_budget=16384,
            request_timeout_seconds=45,
            max_retries=0,
        )
    )

    result = client.complete_json(
        system_prompt="Return an answer.",
        user_prompt="Test",
        schema=_StructuredAnswer,
    )

    assert result.answer == "ok"
    request = calls[0]
    assert request["url"] == "https://integrate.api.nvidia.com/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer nvapi-test"
    assert request["timeout"] == 45
    assert request["json"]["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert request["json"]["temperature"] == 1.0
    assert request["json"]["top_p"] == 0.95
    assert request["json"]["max_tokens"] == 16384
    assert request["json"]["stream"] is False
    assert request["json"]["chat_template_kwargs"] == {"enable_thinking": True}
    assert request["json"]["reasoning_budget"] == 16384


def test_nvidia_nim_can_disable_reasoning_parameters(monkeypatch):
    payloads = []

    def fake_post(_url, *, headers, json, timeout):
        payloads.append(json)
        return _RequestsResponse(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    client = NvidiaNIMLLMClient(
        NvidiaNIMConfig(
            api_key="nvapi-test",
            reasoning_enabled=False,
            max_retries=0,
        )
    )

    assert client.complete_text(system_prompt="System", user_prompt="User") == "ok"
    assert "chat_template_kwargs" not in payloads[0]
    assert "reasoning_budget" not in payloads[0]


def test_nvidia_dns_failure_is_retried_and_returns_actionable_code(monkeypatch):
    calls = []

    def failing_post(_url, *, headers, json, timeout):
        calls.append((_url, timeout))
        raise requests.ConnectionError(
            "NameResolutionError: Failed to resolve 'integrate.api.nvidia.com' "
            "([Errno 11001] getaddrinfo failed)"
        )

    monkeypatch.setattr("app.llm.client.requests.post", failing_post)
    monkeypatch.setattr("app.llm.client.time.sleep", lambda _seconds: None)
    client = NvidiaNIMLLMClient(
        NvidiaNIMConfig(
            api_key="nvapi-test",
            model="minimaxai/minimax-m3",
            max_retries=2,
            retry_backoff_seconds=0,
        )
    )

    with pytest.raises(LLMServiceError, match="NVIDIA_DNS_RESOLUTION_FAILED") as exc:
        client.complete_text(system_prompt="System", user_prompt="User")

    assert len(calls) == 3
    assert "API key were not contacted" in str(exc.value)


def test_nvidia_every_http_attempt_uses_shared_limiter_and_429_penalizes_it(
    monkeypatch,
):
    responses = [
        _RequestsResponse(
            {"status": 429, "title": "Too Many Requests"},
            status_code=429,
            headers={"Retry-After": "7"},
        ),
        _RequestsResponse(
            {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        ),
    ]

    def fake_post(_url, *, headers, json, timeout):
        return responses.pop(0)

    class FakeLimiter:
        def __init__(self):
            self.acquired = 0
            self.penalties = []

        def acquire(self):
            self.acquired += 1
            return SimpleNamespace(
                waited_seconds=0,
                requests_in_window=self.acquired,
                remaining_requests=60 - self.acquired,
                limit=60,
            )

        def penalize(self, seconds):
            self.penalties.append(seconds)

        def status(self):
            return {
                "limit": 60,
                "requests_in_window": self.acquired,
                "remaining_requests": 60 - self.acquired,
                "window_seconds": 60,
                "next_request_in_seconds": 0,
                "cooldown_seconds": 0,
            }

    limiter = FakeLimiter()
    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    monkeypatch.setattr("app.llm.client._nvidia_rate_limiter", lambda _config: limiter)
    client = NvidiaNIMLLMClient(
        NvidiaNIMConfig(
            api_key="nvapi-test",
            model="minimaxai/minimax-m3",
            max_retries=1,
            rate_limit_429_cooldown_seconds=60,
        )
    )

    assert client.complete_text(system_prompt="System", user_prompt="User") == "ok"
    assert limiter.acquired == 2
    assert limiter.penalties == [7]


def test_nvidia_laguna_uses_supplied_profile_and_native_tools(monkeypatch):
    payloads = []

    def fake_post(_url, *, headers, json, timeout):
        payloads.append(json)
        return _RequestsResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_text_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    client = NvidiaNIMLLMClient(
        NvidiaNIMConfig(
            api_key="nvapi-test",
            model="poolside/laguna-xs-2.1",
            max_tokens=16384,
            reasoning_enabled=True,
            max_retries=0,
        )
    )
    message = client.complete_with_tools(
        messages=[{"role": "user", "content": "Read README.md"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_text_file",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert client.supports_tool_calling is True
    assert message["tool_calls"][0]["function"]["name"] == "read_text_file"
    assert payloads[0]["model"] == "poolside/laguna-xs-2.1"
    assert payloads[0]["max_tokens"] == 8192
    assert payloads[0]["tool_choice"] == "auto"
    assert "chat_template_kwargs" not in payloads[0]
    assert "reasoning_budget" not in payloads[0]


@pytest.mark.parametrize(
    ("model", "top_p", "max_tokens", "seed"),
    [
        ("z-ai/glm-5.2", 1.0, 16384, 42),
        ("minimaxai/minimax-m3", 0.95, 8192, None),
    ],
)
def test_new_nvidia_models_use_native_json_schema_and_exact_profiles(
    monkeypatch, model, top_p, max_tokens, seed
):
    payloads = []

    def fake_post(_url, *, headers, json, timeout):
        payloads.append(json)
        return _RequestsResponse(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": '{"answer":"ok"}'}}
                ]
            }
        )

    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    client = NvidiaNIMLLMClient(
        NvidiaNIMConfig(
            api_key="nvapi-test",
            model=model,
            temperature=0.1,
            top_p=0.2,
            max_tokens=128,
            reasoning_enabled=True,
            max_retries=0,
        )
    )
    result = client.complete_json(
        system_prompt="Return an answer.",
        user_prompt="Test",
        schema=_StructuredAnswer,
    )

    assert result.answer == "ok"
    assert client.supports_tool_calling is True
    assert client.supports_structured_output is True
    payload = payloads[0]
    assert payload["model"] == model
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == top_p
    assert payload["max_tokens"] == max_tokens
    assert payload["stream"] is False
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert "chat_template_kwargs" not in payload
    assert "reasoning_budget" not in payload
    if seed is None:
        assert "seed" not in payload
    else:
        assert payload["seed"] == seed


def test_azure_kimi_complete_with_tools_uses_sdk_tool_surface():
    captured = {}

    class Message(dict):
        def as_dict(self):
            return dict(self)

    class FakeAzureClient:
        def complete(self, **kwargs):
            captured.update(kwargs)
            message = Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call-azure",
                        "type": "function",
                        "function": {"name": "list_directory", "arguments": "{}"},
                    }
                ],
            )
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice], "usage": None})()

    client = AzureFoundryLLMClient.__new__(AzureFoundryLLMClient)
    client._config = AzureAIConfig(
        endpoint="https://example.services.ai.azure.com/models",
        api_key="test-key",
        model_deployment="Kimi-K2.6",
    )
    client._client = FakeAzureClient()
    message = client.complete_with_tools(
        messages=[{"role": "user", "content": "List files"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    assert client.supports_tool_calling is True
    assert captured["tool_choice"] == "auto"
    assert captured["model"] == "Kimi-K2.6"
    assert message["tool_calls"][0]["id"] == "call-azure"


def test_openrouter_uses_reasoning_and_strict_json_schema(monkeypatch):
    calls = []

    def fake_post(url, *, headers, json, timeout):
        calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return _RequestsResponse(
            {
                "choices": [{"message": {"role": "assistant", "content": '{"answer":"ok"}'}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        )

    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    client = OpenRouterLLMClient(
        OpenRouterConfig(
            api_key="test-key",
            model="cohere/north-mini-code:free",
            request_timeout_seconds=22,
        )
    )

    result = client.complete_json(
        system_prompt="Return an answer.",
        user_prompt="Test",
        schema=_StructuredAnswer,
    )

    assert result.answer == "ok"
    request = calls[0]
    assert request["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer test-key"
    assert request["timeout"] == 22
    assert request["json"]["model"] == "cohere/north-mini-code:free"
    assert request["json"]["reasoning"] == {"enabled": True}
    assert request["json"]["response_format"]["type"] == "json_schema"
    assert request["json"]["response_format"]["json_schema"]["strict"] is True


def test_openrouter_catalog_parses_requested_model_parameters(monkeypatch):
    captured = {}

    def fake_get(url, *, headers, timeout):
        captured.update(url=url, headers=headers, timeout=timeout)
        return _RequestsResponse(
            {
                "data": [
                    {
                        "id": "liquid/lfm-2.5-2.6b:free",
                        "name": "LiquidAI: LFM2.5-2.6B (free)",
                        "description": "Small reasoning model.",
                        "context_length": 128000,
                        "architecture": {
                            "input_modalities": ["text"],
                            "output_modalities": ["text"],
                        },
                        "supported_parameters": [
                            "temperature",
                            "reasoning",
                            "response_format",
                        ],
                        "pricing": {"prompt": "0", "completion": "0"},
                    },
                    {"id": "not/featured", "name": "Ignored"},
                ]
            }
        )

    monkeypatch.setattr("app.llm.client.requests.get", fake_get)
    details = list_openrouter_model_details(
        OpenRouterConfig(
            api_key="test-key",
            discovery_timeout_seconds=7,
        )
    )

    model = details["liquid/lfm-2.5-2.6b:free"]
    assert model["context_length"] == 128000
    assert model["reasoning_supported"] is True
    assert model["structured_output_supported"] is True
    assert model["input_modalities"] == ["text"]
    assert model["prompt_price"] == "0"
    assert "not/featured" not in details
    assert captured == {
        "url": "https://openrouter.ai/api/v1/models",
        "headers": {
            "Accept": "application/json",
            "Authorization": "Bearer test-key",
        },
        "timeout": 7,
    }


def test_requested_openrouter_models_are_in_featured_selector():
    requested = {
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
    }
    assert requested <= set(OPENROUTER_FEATURED_MODELS)
    assert "cohere/north-mini-code:free" in OPENROUTER_FEATURED_MODELS


def test_openrouter_preserves_reasoning_details_during_json_repair(monkeypatch):
    reasoning_details = [{"type": "reasoning.text", "text": "opaque provider data"}]
    payloads = []
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"wrong":"shape"}',
                            "reasoning_details": reasoning_details,
                        }
                    }
                ]
            },
            {"choices": [{"message": {"role": "assistant", "content": '{"answer":"fixed"}'}}]},
        ]
    )

    def fake_post(_url, *, headers, json, timeout):
        payloads.append(json)
        return _RequestsResponse(next(responses))

    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    client = OpenRouterLLMClient(OpenRouterConfig(api_key="test-key"))

    result = client.complete_json(
        system_prompt="Return an answer.",
        user_prompt="Test",
        schema=_StructuredAnswer,
    )

    assert result.answer == "fixed"
    assistant_turn = payloads[1]["messages"][2]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["reasoning_details"] == reasoning_details


def test_openrouter_retries_429_and_preserves_upstream_provider_detail(monkeypatch):
    calls = []
    sleeps = []

    def fake_post(_url, *, headers, json, timeout):
        calls.append(json)
        return _RequestsResponse(
            {
                "error": {
                    "message": "Provider returned error",
                    "code": 429,
                    "metadata": {
                        "provider_name": "Google AI Studio",
                        "raw": json_module.dumps(
                            {
                                "error": {
                                    "status": "RESOURCE_EXHAUSTED",
                                    "message": "Upstream quota exhausted.",
                                }
                            }
                        ),
                    },
                }
            },
            status_code=429,
            headers={"Retry-After": "0"},
        )

    # The callback parameter is named json to mirror requests.post, so keep a
    # separate module alias for constructing the nested provider payload.
    json_module = json
    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    monkeypatch.setattr("app.llm.client.time.sleep", sleeps.append)
    client = OpenRouterLLMClient(
        OpenRouterConfig(
            api_key="test-key",
            model="google/gemma-4-31b-it:free",
            max_retries=1,
            retry_backoff_seconds=0,
        )
    )

    with pytest.raises(LLMRateLimitError) as exc_info:
        client.complete_text(system_prompt="Be concise.", user_prompt="Hello")

    message = str(exc_info.value)
    assert len(calls) == 2
    assert sleeps == [0.0]
    assert "HTTP 429" in message
    assert "google/gemma-4-31b-it:free" in message
    assert "Google AI Studio" in message
    assert "Upstream quota exhausted" in message


def test_openrouter_transient_429_can_recover_on_retry(monkeypatch):
    responses = iter(
        [
            _RequestsResponse(
                {"error": {"message": "Provider returned error", "code": 429}},
                status_code=429,
                headers={"Retry-After": "0"},
            ),
            _RequestsResponse(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            ),
        ]
    )
    monkeypatch.setattr(
        "app.llm.client.requests.post",
        lambda *_args, **_kwargs: next(responses),
    )
    monkeypatch.setattr("app.llm.client.time.sleep", lambda _seconds: None)
    client = OpenRouterLLMClient(
        OpenRouterConfig(api_key="test-key", max_retries=1, retry_backoff_seconds=0)
    )

    assert client.complete_text(system_prompt="Be concise.", user_prompt="Hello") == "ok"


def test_openrouter_normalizes_intent_aliases_without_a_repair_call(monkeypatch):
    from app.llm.schemas import IntentResult

    payloads = []

    def fake_post(_url, *, headers, json, timeout):
        payloads.append(json)
        return _RequestsResponse(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"classification":"aggregation",'
                                '"relevant_tables":["FactInternetSales"]}'
                            ),
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("app.llm.client.requests.post", fake_post)
    client = OpenRouterLLMClient(OpenRouterConfig(api_key="test-key"))

    result = client.complete_json(
        system_prompt="Classify the question.",
        user_prompt="Total sales?",
        schema=IntentResult,
    )

    assert result.analysis_type == "aggregation"
    assert result.intent_summary
    assert len(payloads) == 1
