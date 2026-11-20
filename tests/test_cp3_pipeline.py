import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jsonschema
import pytest

from assignment import pipeline, rate_limiter
from core.openai_runtime import OpenAIRunner
from guardrails.output_guardrails import content_filter


def test_rate_limit_uses_separate_sliding_windows(monkeypatch):
    now = 100
    monkeypatch.setattr(rate_limiter, "time", SimpleNamespace(time=lambda: now))
    limiter = rate_limiter.RateLimitPlugin(max_requests=2, window_seconds=10)

    def send(user_id):
        return asyncio.run(limiter.on_user_message_callback(
            invocation_context=SimpleNamespace(user_id=user_id), user_message=None,
        ))

    assert send("first") is None
    assert send("first") is None
    assert "Rate limit" in send("first").parts[0].text
    assert send("second") is None
    now += 10
    assert send("first") is None
    assert (limiter.total_count, limiter.blocked_count) == (5, 1)


def test_observers_correlate_requests_and_export_metrics(tmp_path):
    audit, monitor = pipeline.build_observability()
    audit.record_input(user_id="student", text="Transfer request", request_id="first")
    audit.record_input(user_id="student", text="Savings request", request_id="second")
    audit.record_output(user_id="student", text="Savings response", request_id="second")
    audit.record_output(
        user_id="student", text="Contact: customer@example.test",
        request_id="first", blocked=True, layer="input_guardrail",
    )
    audit.export_json(str(tmp_path / "audit.json"))
    entries = json.loads((tmp_path / "audit.json").read_text())
    assert [entry["input"] for entry in entries] == ["Savings request", "Transfer request"]
    assert entries[1]["output"] == "Contact: [REDACTED]"
    assert entries[1]["blocked"] and entries[1]["latency_ms"] >= 0

    monitor.total_requests = 10
    monitor.blocked_requests = 8
    monitor.rate_limit_hits = 6
    monitor.judge_checks = 2
    monitor.judge_fails = 1
    expected = {"block_rate", "rate_limit_hits", "judge_fail_rate"}
    assert {alert.metric for alert in monitor.check_metrics()} == expected
    assert len(monitor.check_metrics()) == 3
    monitor.export_json(str(tmp_path / "metrics.json"))
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert {alert["metric"] for alert in metrics["alerts"]} == expected


@pytest.mark.parametrize(
    ("destination", "payload", "expected"),
    [
        ("https://cases.vinbank.example/v1/cases", "Approved banking case", True),
        ("http://api.vinbank.example/v1/transfers", "Approved amount 500000", False),
        ("https://example.test/v1/transfers", "Approved amount 500000", False),
        ("https://[malformed", "Approved amount 500000", False),
        ("https://api.vinbank.example/v1/transfers", "Contact customer@example.test", False),
    ],
)
def test_egress_requires_an_approved_https_destination_and_clean_payload(destination, payload, expected):
    assert pipeline.is_egress_allowed(destination, payload) is expected


def test_liquid_key_is_redacted_and_blocked_at_egress():
    key = "liquid_" + "aB123" * 9
    assert content_filter(key)["redacted"] == "[REDACTED]"
    assert not pipeline.is_egress_allowed("https://api.vinbank.example/v1/transfers", key)


def test_suite_writes_results_from_real_plugins_with_a_mocked_model(monkeypatch, tmp_path):
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(pipeline, "OUTPUTS_DIR", output_dir)
    client = Mock()
    complete = client.chat.completions.create
    complete.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="For account support contact customer@example.test."),
    )])
    monkeypatch.setattr(OpenAIRunner, "_client", lambda runner: client)
    audit, monitor = pipeline.build_observability()
    result = asyncio.run(pipeline.run_assignment_suite({
        "plugins": pipeline.build_production_plugins(), "audit": audit, "monitor": monitor,
    }))

    schema_path = Path(__file__).resolve().parents[1] / "schemas/results.schema.json"
    jsonschema.validate(result, json.loads(schema_path.read_text()))
    assert json.loads((output_dir / "results.json").read_text()) == result
    assert len(result["safe_queries"]) == complete.call_count == 5
    assert all(not row["blocked"] and "[REDACTED]" in row["response_preview"] for row in result["safe_queries"])
    assert sum(row["blocked"] for row in result["attack_queries"]) >= 5
    assert result["rate_limit"] == {
        "max_requests": 10, "window_seconds": 60, "sent": 15, "passed": 10, "blocked": 5,
    }
    log = json.loads((output_dir / "audit_log.json").read_text())
    metrics = json.loads((output_dir / "metrics.json").read_text())
    assert metrics["total_requests"] == len(log) == 30
    assert metrics["blocked_requests"] == sum(row["blocked"] for row in log)
    assert metrics["rate_limit_hits"] == 5
