from __future__ import annotations

from deepresearch.config import ResearchConfig
from deepresearch.core.llm import LLMInvocation, LLMWorkers
from deepresearch.core.payloads import PlannerPayload
from deepresearch.telemetry import TelemetryRecorder


def _planner_variables() -> dict[str, object]:
    return {
        "query": "What is Lightpanda?",
        "has_subqueries": False,
        "coverage_summary": "- No subqueries yet.",
        "source_balance_summary": "- No evidence has been accepted yet.",
        "active_subqueries": "- None",
        "resolved_subqueries": "- None",
        "open_gaps": "- None",
        "dossier_context": "",
    }


def test_telemetry_recorder_filters_by_verbosity() -> None:
    recorder = TelemetryRecorder(verbosity=1)

    included = recorder.record("planner", "Agenda updated", verbosity=1, payload_type="decision", count=1)
    skipped = recorder.record("planner", "LLM response parsed", verbosity=2, payload_type="llm_response", count=1)

    assert included is not None
    assert included.payload_type == "decision"
    assert skipped is None


def test_telemetry_recorder_normalizes_long_payloads() -> None:
    recorder = TelemetryRecorder(verbosity=2)

    event = recorder.record(
        "planner",
        "LLM response parsed",
        verbosity=2,
        payload_type="llm_response",
        raw_output="x" * 5000,
    )

    assert event is not None
    assert isinstance(event.payload["raw_output"], str)
    assert len(event.payload["raw_output"]) < 1500


def test_llmworkers_records_successful_llm_response_events(monkeypatch, tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    recorder = TelemetryRecorder(verbosity=2)
    workers = LLMWorkers(config, telemetry=recorder)

    monkeypatch.setattr(workers, "_llm", lambda temperature, json_format=True: object())
    monkeypatch.setattr(
        workers,
        "_invoke",
        lambda llm, messages: LLMInvocation(
            content='{"subqueries": [{"question": "Capabilities", "rationale": "Understand product", "search_terms": ["lightpanda capabilities"]}], "search_intents": [], "hypotheses": ["Lightpanda is a browser"]}',
            usage={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        ),
    )

    payload, usage = workers._parse_response("planner", _planner_variables(), PlannerPayload, 0.2)
    events = workers.consume_telemetry_events()

    assert payload.subqueries
    assert usage["total_tokens"] == 19
    assert len(events) == 1
    assert events[0].payload_type == "llm_response"
    assert events[0].payload["repair_attempted"] is False
    assert "parsed_output" in events[0].payload


def test_llmworkers_records_repair_flow_events(monkeypatch, tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    recorder = TelemetryRecorder(verbosity=2)
    workers = LLMWorkers(config, telemetry=recorder)
    responses = iter([
        LLMInvocation(content="not valid json", usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}),
        LLMInvocation(
            content='{"subqueries": [{"question": "Limitations", "rationale": "Need weaknesses", "search_terms": ["lightpanda limitations"]}], "search_intents": [], "hypotheses": []}',
            usage={"input_tokens": 14, "output_tokens": 9, "total_tokens": 23},
        ),
    ])

    monkeypatch.setattr(workers, "_llm", lambda temperature, json_format=True: object())
    monkeypatch.setattr(workers, "_invoke", lambda llm, messages: next(responses))

    payload, usage = workers._parse_response("planner", _planner_variables(), PlannerPayload, 0.2)
    events = workers.consume_telemetry_events()

    assert payload.subqueries[0].question == "Limitations"
    assert usage["total_tokens"] == 23
    assert len(events) == 2
    assert events[0].payload_type == "llm_repair"
    assert events[1].payload_type == "llm_response"
    assert events[1].payload["repair_attempted"] is True
