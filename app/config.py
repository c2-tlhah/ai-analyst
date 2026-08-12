"""Centralized configuration loaded from environment variables / .env.

Every credential and tunable limit used by the backend flows through this
module so there is a single, auditable source of truth. Nothing here talks
to the network or the database -- it only parses configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load a .env file if present. Real environment variables always win.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def _normalize_inference_endpoint(raw: str) -> str:
    """Normalize an Azure AI Foundry endpoint to the Model Inference API root.

    The Foundry portal often surfaces a *project* endpoint of the form
    ``https://<resource>.services.ai.azure.com/api/projects/<project>``
    (for the ``azure-ai-projects`` SDK / agents). The Azure AI Model
    Inference API used by :mod:`app.llm.client` instead lives at
    ``https://<resource>.services.ai.azure.com/models``. If a project-style
    URL is supplied, rewrite it to the inference root automatically so
    either form works when pasted into ``.env``.
    """
    if not raw:
        return raw
    if "/api/projects/" in raw:
        scheme_and_host = raw.split("/api/projects/", 1)[0]
        return f"{scheme_and_host.rstrip('/')}/models"
    return raw


@dataclass(frozen=True)
class AzureAIConfig:
    endpoint: str = field(
        default_factory=lambda: _normalize_inference_endpoint(
            os.getenv("AZURE_FOUNDRY_ENDPOINT", "")
        )
    )
    api_key: str = field(default_factory=lambda: os.getenv("AZURE_FOUNDRY_API_KEY", ""))
    model_deployment: str = field(
        default_factory=lambda: os.getenv("AZURE_FOUNDRY_MODEL", "Kimi-K2.6")
    )
    api_version: str = field(
        default_factory=lambda: os.getenv("AZURE_FOUNDRY_API_VERSION", "")
    )
    temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.1))
    # Kimi K2.6 is a reasoning model: it spends tokens on a hidden
    # `reasoning_content` chain-of-thought *before* the final `content`.
    # Too small a budget truncates generation before any answer is emitted
    # (finish_reason="length" with empty content) -- so the default here is
    # generous relative to a typical non-reasoning chat model.
    max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 4096))
    request_timeout_seconds: int = field(
        default_factory=lambda: _get_int("AZURE_FOUNDRY_REQUEST_TIMEOUT_SECONDS", 120)
    )

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.api_key)


@dataclass(frozen=True)
class OllamaConfig:
    """Connection and generation settings for a local Ollama server."""

    base_url: str = field(
        default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    )
    model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", ""))
    temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 4096))
    request_timeout_seconds: int = field(
        # A local model may need to load several GB and process a sizeable schema
        # prompt on CPU. Keep this above the Foundry timeout by default.
        default_factory=lambda: _get_int("OLLAMA_REQUEST_TIMEOUT_SECONDS", 300)
    )
    discovery_timeout_seconds: int = field(
        default_factory=lambda: _get_int("OLLAMA_DISCOVERY_TIMEOUT_SECONDS", 3)
    )
    auto_start: bool = field(
        default_factory=lambda: _get_bool("OLLAMA_AUTO_START", True)
    )
    executable: str = field(
        default_factory=lambda: os.getenv("OLLAMA_EXECUTABLE", "").strip()
    )
    startup_timeout_seconds: int = field(
        default_factory=lambda: _get_int("OLLAMA_STARTUP_TIMEOUT_SECONDS", 20)
    )
    stop_on_exit: bool = field(
        default_factory=lambda: _get_bool("OLLAMA_STOP_ON_EXIT", True)
    )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)


@dataclass(frozen=True)
class OpenRouterConfig:
    """OpenRouter chat-completions configuration."""

    base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/")
    )
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    model: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_MODEL", "cohere/north-mini-code:free"
        )
    )
    reasoning_enabled: bool = field(
        default_factory=lambda: _get_bool("OPENROUTER_REASONING_ENABLED", True)
    )
    temperature: float = field(default_factory=lambda: _get_float("LLM_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _get_int("LLM_MAX_TOKENS", 4096))
    request_timeout_seconds: int = field(
        default_factory=lambda: _get_int("OPENROUTER_REQUEST_TIMEOUT_SECONDS", 180)
    )
    http_referer: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_HTTP_REFERER", "")
    )
    app_title: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_APP_TITLE", "AI Analyst")
    )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("AI_ANALYST_DB_PATH", "data/ai_analyst.db"))
    )


@dataclass(frozen=True)
class MetadataConfig:
    directory: Path = field(
        default_factory=lambda: _resolve_path(
            os.getenv("AI_ANALYST_METADATA_DIR", "metadata_store")
        )
    )

    @property
    def schema_file(self) -> Path:
        return self.directory / "schema_metadata.json"

    @property
    def business_context_file(self) -> Path:
        return self.directory / "business_context.json"


@dataclass(frozen=True)
class QueryLimits:
    max_rows: int = field(default_factory=lambda: _get_int("SQL_MAX_ROWS", 5000))
    download_max_rows: int = field(
        default_factory=lambda: _get_int("SQL_DOWNLOAD_MAX_ROWS", 250000)
    )
    statement_timeout_seconds: int = field(
        default_factory=lambda: _get_int("SQL_STATEMENT_TIMEOUT_SECONDS", 15)
    )
    max_retries: int = field(default_factory=lambda: _get_int("SQL_MAX_RETRIES", 2))


@dataclass(frozen=True)
class LoggingConfig:
    level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    file: Path = field(
        default_factory=lambda: _resolve_path(os.getenv("LOG_FILE", "logs/ai_analyst.log"))
    )


@dataclass(frozen=True)
class UIConfig:
    preview_rows: int = field(
        default_factory=lambda: _get_int("UI_PREVIEW_ROWS", 200)
    )


@dataclass(frozen=True)
class Settings:
    azure_ai: AzureAIConfig = field(default_factory=AzureAIConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    default_llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_DEFAULT_PROVIDER", "azure_foundry")
    )
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)
    limits: QueryLimits = field(default_factory=QueryLimits)
    ui: UIConfig = field(default_factory=UIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide settings singleton (lazily constructed)."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
