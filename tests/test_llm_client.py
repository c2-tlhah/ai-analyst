import json

import pytest
from pydantic import BaseModel

from app.config import OllamaConfig, OpenRouterConfig
from app.llm.client import (
    LLMError,
    OllamaLLMClient,
    OpenRouterLLMClient,
    _foundry_base_url,
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
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
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
