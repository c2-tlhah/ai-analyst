from app.llm.presentation import (
    operation_status,
    provider_label,
    provider_runtime_note,
)


def test_openrouter_status_names_selected_provider_and_model_only():
    message = operation_status(
        "openrouter", "google/gemma-4-31b-it:free", "data"
    )

    assert "OpenRouter" in message
    assert "google/gemma-4-31b-it:free" in message
    assert "Free route" in message
    assert "NVIDIA" not in message
    assert "rolling-minute" not in message


def test_ollama_status_explains_local_model_loading():
    message = operation_status("ollama", "qwen2.5:7b", "knowledge")

    assert "Ollama (local)" in message
    assert "qwen2.5:7b" in message
    assert "this machine" in message
    assert "CPU/GPU" in message


def test_nvidia_status_is_the_only_one_that_mentions_local_queueing():
    nvidia = provider_runtime_note("nvidia_nim", "z-ai/glm-5.2")
    azure = provider_runtime_note("azure_foundry", "Kimi-K2.6")

    assert "queued locally" in nvidia
    assert "queued locally" not in azure


def test_all_provider_labels_are_consistent():
    assert provider_label("azure_foundry") == "Azure AI Foundry"
    assert provider_label("ollama") == "Ollama (local)"
    assert provider_label("openrouter") == "OpenRouter"
    assert provider_label("nvidia_nim") == "NVIDIA NIM"
