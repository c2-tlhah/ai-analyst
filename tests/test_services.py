from pathlib import Path

import pytest

from app.config import OllamaConfig
from app import services


class FakeProcess:
    pid = 1234

    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


@pytest.fixture(autouse=True)
def _reset_managed_process():
    services.stop_managed_services()
    yield
    services.stop_managed_services()


def test_existing_ollama_is_reused_without_starting(monkeypatch):
    monkeypatch.setattr(services, "_probe_ollama", lambda _config: True)
    monkeypatch.setattr(
        services.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("Ollama should not be started twice"),
    )

    status = services.ensure_ollama_running(OllamaConfig())

    assert status.running is True
    assert status.started_by_app is False


def test_disabled_auto_start_does_not_launch(monkeypatch):
    monkeypatch.setattr(services, "_probe_ollama", lambda _config: False)
    config = OllamaConfig(auto_start=False)

    status = services.ensure_ollama_running(config)

    assert status.running is False
    assert "disabled" in status.message


def test_remote_ollama_is_never_started_as_local_process(monkeypatch):
    monkeypatch.setattr(services, "_probe_ollama", lambda _config: False)
    config = OllamaConfig(base_url="https://ollama.example.test")

    status = services.ensure_ollama_running(config)

    assert status.running is False
    assert "Remote endpoints" in status.message


def test_local_ollama_is_started_and_waited_until_ready(monkeypatch, tmp_path):
    probes = iter([False, True])
    process = FakeProcess()
    captured = {}

    monkeypatch.setattr(services, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(services, "_probe_ollama", lambda _config: next(probes))
    monkeypatch.setattr(
        services,
        "_find_ollama_executable",
        lambda _config: Path("C:/Program Files/Ollama/ollama.exe"),
    )

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(services.subprocess, "Popen", fake_popen)

    status = services.ensure_ollama_running(
        OllamaConfig(startup_timeout_seconds=1)
    )

    assert status.running is True
    assert status.started_by_app is True
    assert captured["args"][-1] == "serve"
    assert captured["kwargs"]["env"]["OLLAMA_HOST"] == "localhost:11434"

    services.stop_managed_services()
    assert process.terminated is True
