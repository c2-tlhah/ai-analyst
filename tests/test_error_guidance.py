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
