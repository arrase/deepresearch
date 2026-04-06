from __future__ import annotations

from deepresearch.config import ResearchConfig
from deepresearch.context_manager import NodeContext
from deepresearch.core.llm import LLMInvocation, LLMWorkers
from deepresearch.core.payloads import PlannerPayload


def _planner_variables() -> dict[str, object]:
    return {
        "query": "What is Tavily?",
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
                '"Understand product", "search_terms": ["tavily capabilities"]}], '
                '"search_intents": [], "hypotheses": ["Tavily is a research API"]}'
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
                '"Need weaknesses", "search_terms": ["tavily limitations"]}], '
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


def test_synthesize_report_parses_semantic_sections(monkeypatch, tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    workers = LLMWorkers(config)

    monkeypatch.setattr(workers, "_llm", lambda temperature, json_format=True: object())
    monkeypatch.setattr(
        workers,
        "_invoke",
        lambda llm, messages: LLMInvocation(
            content=(
                "# Tavily\n\n"
                "## Executive Answer\nTavily is a research API.\n\n"
                "## Key Findings\n- It returns search results.\n- It can include raw content.\n\n"
                "## Analysis\nTavily supports research workflows.\n\n"
                "## Limitations And Open Gaps\n- Local benchmarks remain limited."
            ),
            usage={"input_tokens": 12, "output_tokens": 20, "total_tokens": 32},
        ),
    )

    report, usage = workers.synthesize_report_with_usage(
        NodeContext(query="What is Tavily?"),
        query="What is Tavily?",
    )

    assert report.executive_answer == "Tavily is a research API."
    assert report.key_findings == ["It returns search results.", "It can include raw content."]
    assert report.open_gaps == ["Local benchmarks remain limited."]
    assert usage["total_tokens"] == 32


def test_synthesize_report_parses_spanish_sections(monkeypatch, tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    workers = LLMWorkers(config)

    monkeypatch.setattr(workers, "_llm", lambda temperature, json_format=True: object())
    monkeypatch.setattr(
        workers,
        "_invoke",
        lambda llm, messages: LLMInvocation(
            content=(
                "# OpenAI\n\n"
                "## Respuesta Ejecutiva\nOpenAI podría quebrar si no mantiene acceso constante a capital externo.\n\n"
                "## Hallazgos Clave\n"
                "- Las pérdidas son muy elevadas.\n"
                "- La financiación externa sigue siendo crítica.\n\n"
                "## Análisis\nExiste presión financiera y competitiva.\n\n"
                "## Limitaciones Y Brechas Abiertas\n- Faltan comparativas directas con competidores."
            ),
            usage={"input_tokens": 15, "output_tokens": 24, "total_tokens": 39},
        ),
    )

    report, usage = workers.synthesize_report_with_usage(
        NodeContext(query="puede quebrar Open AI?"),
        query="puede quebrar Open AI?",
    )

    assert report.executive_answer == "OpenAI podría quebrar si no mantiene acceso constante a capital externo."
    assert report.key_findings == ["Las pérdidas son muy elevadas.", "La financiación externa sigue siendo crítica."]
    assert report.open_gaps == ["Faltan comparativas directas con competidores."]
    assert usage["total_tokens"] == 39


def test_synthesize_report_parses_sections_by_order_for_arbitrary_language(monkeypatch, tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    workers = LLMWorkers(config)

    monkeypatch.setattr(workers, "_llm", lambda temperature, json_format=True: object())
    monkeypatch.setattr(
        workers,
        "_invoke",
        lambda llm, messages: LLMInvocation(
            content=(
                "# Rapport\n\n"
                "## Synthese\nOpenAI peut faire faillite si son acces au capital se deteriore fortement.\n\n"
                "## Points principaux\n"
                "- Les pertes sont elevees.\n"
                "- La dependance au financement externe reste forte.\n\n"
                "## Analyse\nLa pression concurrentielle et financiere reste importante.\n\n"
                "## Lacunes\n- Il manque une comparaison chiffrée avec les concurrents."
            ),
            usage={"input_tokens": 17, "output_tokens": 21, "total_tokens": 38},
        ),
    )

    report, usage = workers.synthesize_report_with_usage(
        NodeContext(query="OpenAI peut-elle faire faillite ?"),
        query="OpenAI peut-elle faire faillite ?",
    )

    assert report.executive_answer == "OpenAI peut faire faillite si son acces au capital se deteriore fortement."
    assert report.key_findings == ["Les pertes sont elevees.", "La dependance au financement externe reste forte."]
    assert report.open_gaps == ["Il manque une comparaison chiffrée avec les concurrents."]
    assert usage["total_tokens"] == 38
