import json
from dataclasses import dataclass

from app.logging_config import redact_log_text
from app.observability import emit_trace, get_recent_trace_events, trace_span, traced_operation


def test_trace_span_keeps_nested_events_under_one_trace_and_redacts_secrets():
    secret = "nvapi-this-must-never-appear"
    with trace_span("test_request", metadata={"api_key": secret}) as trace_id:
        emit_trace(
            "provider_call",
            category="llm",
            status="completed",
            message=f"Authorization: Bearer {secret}",
            metadata={"token": secret, "model": "test-model"},
        )

    events = get_recent_trace_events(trace_id=trace_id)
    serialized = json.dumps(events)
    assert len(events) == 3
    assert {event["trace_id"] for event in events} == {trace_id}
    assert secret not in serialized
    assert "[REDACTED]" in serialized


def test_traced_operation_attaches_trace_id_to_handled_response():
    @dataclass
    class Response:
        status: str = "ok"
        trace_id: str | None = None

    @traced_operation("unit_operation")
    def run() -> Response:
        return Response()

    response = run()
    assert response.trace_id
    assert get_recent_trace_events(trace_id=response.trace_id)


def test_readable_log_redaction_preserves_format_values():
    assert redact_log_text("Authorization: Bearer nvapi-secret-value") == (
        "Authorization: Bearer [REDACTED]"
    )
    assert "sk-secret-value" not in redact_log_text("api_key=sk-secret-value")
