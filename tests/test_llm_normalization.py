from deepresearch.config import ResearchConfig
from deepresearch.subagents.llm import LLMWorkers, PlannerPayload, _salvage_evidence_payload


def test_normalize_planner_payload_handles_qwen_style_types() -> None:
    config = ResearchConfig()
    workers = LLMWorkers(config)
    payload = {
        "subqueries": [
            {
                "question": "What is Lightpanda?",
                "rationale": "Need scope",
                "priority": 1,
                "evidence_target": "at least 2 sources",
                "success_criteria": "Find two specific advantages",
                "search_terms": "Lightpanda advantages",
            }
        ],
        "search_intents": [
            {
                "query": "Lightpanda Chromium comparison",
                "rationale": "Need comparison",
                "subquery_ids": [1],
            }
        ],
        "hypotheses": "Lightpanda uses less memory",
    }
    normalized = workers._normalize_payload(PlannerPayload, payload)
    assert normalized["subqueries"][0]["id"] == "sq_plan_1"
    assert normalized["subqueries"][0]["evidence_target"] == 2
    assert normalized["subqueries"][0]["success_criteria"] == ["Find two specific advantages"]
    assert normalized["search_intents"][0]["subquery_ids"] == ["sq_plan_1"]
    assert normalized["hypotheses"] == ["Lightpanda uses less memory"]


def test_salvage_evidence_payload_recovers_complete_objects_from_truncated_json() -> None:
        raw = '''```json
{
    "evidences": [
        {
            "summary": "A",
            "claim": "B",
            "quotation": "C",
            "citation_locator": "L1",
            "relevance_score": 0.9,
            "confidence": "high",
            "caveats": [],
            "tags": ["x"]
        },
        {
            "summary": "D",
            "claim": "E",
            "quotation": "F",
            "citation_locator": "L2",
            "relevance_score": 0.8,
            "confidence": "medium",
            "caveats": []
'''
        salvaged = _salvage_evidence_payload(raw)
        assert salvaged is not None
        assert len(salvaged["evidences"]) == 1
        assert salvaged["evidences"][0]["claim"] == "B"
