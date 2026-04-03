from __future__ import annotations

from deepresearch.config import ResearchConfig
from deepresearch.core.llm import LLMInvocation, LLMWorkers
from deepresearch.core.payloads import PlannerPayload


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


def test_llmworkers_parse_response_returns_usage(monkeypatch, tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    workers = LLMWorkers(config)

    monkeypatch.setattr(workers, "_llm", lambda temperature, json_format=True: object())
    monkeypatch.setattr(
        workers,
        "_invoke",
        lambda llm, messages: LLMInvocation(
            content=(
                '{"subqueries": [{"question": "Capabilities", "rationale": '
                '"Understand product", "search_terms": ["lightpanda capabilities"]}], '
                '"search_intents": [], "hypotheses": ["Lightpanda is a browser"]}'
            ),
            usage={"input_tokens": 12, "output_tokens": 7, "total_tokens": 19},
        ),
    )

    payload, usage = workers._parse_response("planner", _planner_variables(), PlannerPayload, 0.2)

    assert payload.subqueries
    assert usage["total_tokens"] == 19


def test_llmworkers_repair_flow_returns_repaired_payload(monkeypatch, tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    workers = LLMWorkers(config)
    responses = iter([
        LLMInvocation(content="not valid json", usage={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13}),
        LLMInvocation(
            content=(
                '{"subqueries": [{"question": "Limitations", "rationale": '
                '"Need weaknesses", "search_terms": ["lightpanda limitations"]}], '
                '"search_intents": [], "hypotheses": []}'
            ),
            usage={"input_tokens": 14, "output_tokens": 9, "total_tokens": 23},
        ),
    ])

    monkeypatch.setattr(workers, "_llm", lambda temperature, json_format=True: object())
    monkeypatch.setattr(workers, "_invoke", lambda llm, messages: next(responses))

    payload, usage = workers._parse_response("planner", _planner_variables(), PlannerPayload, 0.2)

    assert payload.subqueries[0].question == "Limitations"
    assert usage["total_tokens"] == 23
