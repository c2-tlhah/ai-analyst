"""Lifecycle management for local services used by the application.

Streamlit reruns its page script after every interaction, so service startup
must be idempotent. This module probes Ollama before starting it, serializes
concurrent attempts, and only stops a process that this application launched.
"""

from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from app.config import PROJECT_ROOT, OllamaConfig, get_settings
from app.llm.client import LLMError, list_ollama_models
from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ServiceStatus:
    """User-safe snapshot of a managed service's state."""

    name: str
    running: bool
    started_by_app: bool = False
    message: str = ""


_service_lock = threading.Lock()
_managed_ollama_process: subprocess.Popen | None = None
_ollama_log_handle = None


def _probe_ollama(config: OllamaConfig) -> bool:
    try:
        list_ollama_models(config)
        return True
    except LLMError:
        return False


def _is_local_endpoint(base_url: str) -> bool:
    host = (urlsplit(base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _find_ollama_executable(config: OllamaConfig) -> Path | None:
    """Find Ollama from explicit config, PATH, or common install locations."""
    candidates: list[Path] = []
    if config.executable:
        candidates.append(Path(config.executable).expanduser())

    on_path = shutil.which("ollama")
    if on_path:
        candidates.append(Path(on_path))

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe")

    candidates.extend(
        [
            Path("/Applications/Ollama.app/Contents/Resources/ollama"),
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _close_log_handle() -> None:
    global _ollama_log_handle
    if _ollama_log_handle is not None:
        try:
            _ollama_log_handle.close()
        finally:
            _ollama_log_handle = None


def ensure_ollama_running(
    config: OllamaConfig | None = None,
    *,
    force_start: bool = False,
) -> ServiceStatus:
    """Ensure a local Ollama API is ready, starting ``ollama serve`` if needed."""
    global _managed_ollama_process, _ollama_log_handle

    config = config or get_settings().ollama
    with _service_lock:
        if _probe_ollama(config):
            managed = bool(
                _managed_ollama_process is not None
                and _managed_ollama_process.poll() is None
            )
            return ServiceStatus(
                name="Ollama",
                running=True,
                started_by_app=managed,
                message="Ollama is ready.",
            )

        if not config.auto_start and not force_start:
            return ServiceStatus(
                name="Ollama",
                running=False,
                message="Ollama auto-start is disabled by OLLAMA_AUTO_START.",
            )
        if not _is_local_endpoint(config.base_url):
            return ServiceStatus(
                name="Ollama",
                running=False,
                message=(
                    f"Ollama at {config.base_url} is unavailable. Remote endpoints "
                    "are never started as local processes."
                ),
            )

        if _managed_ollama_process is not None:
            if _managed_ollama_process.poll() is None:
                # Another caller already launched it; continue into the readiness wait.
                process = _managed_ollama_process
            else:
                _managed_ollama_process = None
                _close_log_handle()
                process = None
        else:
            process = None

        if process is None:
            executable = _find_ollama_executable(config)
            if executable is None:
                return ServiceStatus(
                    name="Ollama",
                    running=False,
                    message=(
                        "Ollama is not running and its executable was not found. "
                        "Install Ollama or set OLLAMA_EXECUTABLE."
                    ),
                )

            log_path = PROJECT_ROOT / "logs" / "ollama-service.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            _ollama_log_handle = log_path.open("a", encoding="utf-8")
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            child_env = os.environ.copy()
            parsed_url = urlsplit(config.base_url)
            if parsed_url.netloc:
                child_env["OLLAMA_HOST"] = parsed_url.netloc

            try:
                process = subprocess.Popen(  # noqa: S603 - executable is resolved, args are fixed
                    [str(executable), "serve"],
                    cwd=PROJECT_ROOT,
                    env=child_env,
                    stdin=subprocess.DEVNULL,
                    stdout=_ollama_log_handle,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                    start_new_session=os.name != "nt",
                )
            except OSError as exc:
                _close_log_handle()
                logger.exception("Could not start Ollama")
                return ServiceStatus(
                    name="Ollama",
                    running=False,
                    message=f"Could not start Ollama: {exc}",
                )
            _managed_ollama_process = process
            logger.info("Started Ollama service (PID %s)", process.pid)

        deadline = time.monotonic() + max(1, config.startup_timeout_seconds)
        while time.monotonic() < deadline:
            if _probe_ollama(config):
                return ServiceStatus(
                    name="Ollama",
                    running=True,
                    started_by_app=True,
                    message="Ollama was started automatically and is ready.",
                )
            if process.poll() is not None:
                exit_code = process.returncode
                _managed_ollama_process = None
                _close_log_handle()
                return ServiceStatus(
                    name="Ollama",
                    running=False,
                    message=(
                        f"Ollama exited during startup (code {exit_code}). "
                        "See logs/ollama-service.log."
                    ),
                )
            time.sleep(0.25)

        return ServiceStatus(
            name="Ollama",
            running=False,
            started_by_app=True,
            message=(
                f"Ollama did not become ready within {config.startup_timeout_seconds}s. "
                "It may still be starting; see logs/ollama-service.log."
            ),
        )


def stop_managed_services() -> None:
    """Stop only service processes launched by this application."""
    global _managed_ollama_process

    with _service_lock:
        process = _managed_ollama_process
        _managed_ollama_process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        _close_log_handle()


def _stop_services_at_exit() -> None:
    if get_settings().ollama.stop_on_exit:
        stop_managed_services()


atexit.register(_stop_services_at_exit)
