from app.error_guidance import explain_error


def test_unauthorized_sql_table_is_a_validation_error_not_authentication():
    guidance = explain_error(
        "Unauthorized table referenced: SecretTable.", stage="validation"
    )

    assert guidance.stage == "validation"
    assert "read-only query" in guidance.title
    assert guidance.suggestions


def test_provider_authentication_error_has_configuration_actions():
    guidance = explain_error(
        "OpenRouter request failed: 401 Unauthorized", stage="provider"
    )

    assert guidance.title == "LLM authentication failed"
    assert any("API key" in suggestion for suggestion in guidance.suggestions)


def test_openrouter_rate_limit_recommends_another_model():
    guidance = explain_error(
        "OpenRouter rate limit for google/gemma-4-31b-it:free: HTTP 429; provider: Google",
        stage="workflow",
    )

    assert guidance.stage == "provider"
    assert "OpenRouter model" in guidance.title
    assert any("another OpenRouter model" in suggestion for suggestion in guidance.suggestions)


def test_download_database_failure_keeps_download_context():
    guidance = explain_error("Database error: interrupted", stage="download")

    assert guidance.stage == "download"
    assert "CSV" in guidance.title
    assert any("visible analysis" in suggestion for suggestion in guidance.suggestions)


def test_ollama_missing_executable_has_install_suggestion():
    guidance = explain_error(
        "Ollama is not running and its executable was not found.", stage="service"
    )

    assert guidance.stage == "service"
    assert any("Install Ollama" in suggestion for suggestion in guidance.suggestions)


def test_nvidia_dns_failure_has_specific_network_diagnostics():
    guidance = explain_error(
        "NVIDIA NIM request failed: NVIDIA_DNS_RESOLUTION_FAILED getaddrinfo failed",
        stage="workflow",
    )

    assert guidance.stage == "network"
    assert "DNS" in guidance.title
    assert any("Resolve-DnsName" in suggestion for suggestion in guidance.suggestions)
    assert any("flushdns" in suggestion for suggestion in guidance.suggestions)


def test_nvidia_rate_limit_explains_automatic_queue_and_live_budget():
    guidance = explain_error(
        "NVIDIA NIM local rate-limit request queue timed out", stage="provider"
    )

    assert "NVIDIA request budget" in guidance.title
    assert any("spaces calls" in suggestion for suggestion in guidance.suggestions)
    assert any("Live agent logs" in suggestion for suggestion in guidance.suggestions)
