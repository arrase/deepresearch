from deepresearch.config import ResearchConfig
from deepresearch.context_manager import ContextManager
from deepresearch.graph import build_graph
from deepresearch.nodes import ResearchRuntime
from deepresearch.core.llm import PlannerPayload, EvidencePayload, EvidenceDraft, CoveragePayload
from deepresearch.state import (
    BrowserPageStatus,
    SourceVisit,
    ConfidenceLevel,
    FinalReport,
    SearchCandidate,
    SearchIntent,
    Subquery,
    build_initial_state,
)
from deepresearch.telemetry import TelemetryRecorder


class FakeSearchClient:
    def search(self, query: str, *, max_results: int | None = None):
        return [
            SearchCandidate(
                url="https://example.com/report",
                title="Example report",
                snippet=f"Result for {query}",
                domain="example.com",
            )
        ]


class FakeBrowser:
    def fetch(self, url: str) -> SourceVisit:
        return SourceVisit(
            url=url,
            final_url=url,
            status=BrowserPageStatus.USEFUL,
            title="Example report",
            content="# Example report\nFusion demand is rising in 2026.",
            excerpt="Fusion demand is rising in 2026.",
            exit_code=0,
        )


class FailIfCalledBrowser:
    def fetch(self, url: str) -> SourceVisit:
        raise AssertionError(f"Browser should not be called for url={url}")


class EmptySearchClient:
    def search(self, query: str, *, max_results: int | None = None):
        return []


class FakeLLMWorkers:
    def plan_research(self, context) -> PlannerPayload:
        subquery = Subquery(
            id="sq_demo",
            question="What happened to fusion demand in 2026?",
            rationale="Need primary claim",
            evidence_target=1,
            search_terms=["fusion demand 2026"],
        )
        return PlannerPayload(
            subqueries=[subquery],
            search_intents=[SearchIntent(query="fusion demand 2026", rationale="primary", subquery_ids=[subquery.id])],
            hypotheses=["Demand increased"],
        )

    def plan_research_with_usage(self, context):
        return self.plan_research(context), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    def extract_evidence(self, context) -> EvidencePayload:
        draft = EvidenceDraft(
            summary="Demand increased",
            claim="Fusion demand is rising in 2026.",
            quotation="Fusion demand is rising in 2026.",
            citation_locator="paragraph 1",
            relevance_score=0.9,
            confidence=ConfidenceLevel.HIGH,
            caveats=[],
            tags=["trend"],
        )
        return EvidencePayload(evidences=[draft])

    def extract_evidence_with_usage(self, context):
        return self.extract_evidence(context), {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}

    def evaluate_coverage(self, context) -> CoveragePayload:
        return CoveragePayload(
            resolved_subquery_ids=["sq_demo"],
            contradictions=[],
            open_gaps=[],
            is_sufficient=True,
            rationale="Enough evidence",
        )

    def evaluate_coverage_with_usage(self, context):
        return self.evaluate_coverage(context), {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}

    def synthesize_report(self, context, query: str):
        return FinalReport(
            query=query,
            executive_answer="Fusion demand increased in 2026 according to the accepted source.",
            key_findings=["Demand increased in 2026"],
            confidence=ConfidenceLevel.HIGH,
            reservations=[],
            open_gaps=[],
            cited_sources=[{"url": "https://example.com/report", "title": "Example report", "evidence_ids": ["ev1"]}],
            evidence_ids=["ev1"],
            markdown_report="# Research Report\n\nFusion demand is rising.",
        )

    def synthesize_report_with_usage(self, context, query: str):
        return self.synthesize_report(context, query), {"input_tokens": 30, "output_tokens": 40, "total_tokens": 70}


class InsufficientLLMWorkers(FakeLLMWorkers):
    def evaluate_coverage(self, context) -> CoveragePayload:
        return CoveragePayload(
            resolved_subquery_ids=[],
            contradictions=[],
            open_gaps=[],
            is_sufficient=False,
            rationale="Need more evidence",
        )


class FinalContextFullLLMWorkers(InsufficientLLMWorkers):
    def extract_evidence(self, context) -> EvidencePayload:
        draft = EvidenceDraft(
            summary="Demand increased and costs remain high. " * 8,
            claim="Fusion demand is rising in 2026 while costs remain high.",
            quotation=("Fusion demand is rising in 2026 while costs remain high. " * 12).strip(),
            citation_locator="paragraph 1",
            relevance_score=0.9,
            confidence=ConfidenceLevel.HIGH,
            caveats=[],
            tags=["trend"],
        )
        return EvidencePayload(evidences=[draft])


class ExhaustedLLMWorkers(InsufficientLLMWorkers):
    def extract_evidence(self, context) -> EvidencePayload:
        return EvidencePayload(evidences=[])


def test_graph_runs_end_to_end_with_fakes() -> None:
    config = ResearchConfig()
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=FakeLLMWorkers(),
        browser=FakeBrowser(),
        search_client=FakeSearchClient(),
        telemetry=TelemetryRecorder(),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "What is happening to fusion demand?",
        max_iterations=4,
    )
    result = graph.invoke(initial_state)
    assert result["final_report"] is not None
    assert result["final_report"].executive_answer.startswith("Fusion demand increased")
    assert result["final_report"].markdown_report.startswith("# Research Report")
    assert result["atomic_evidence"]
    assert result["final_report"].stop_reason == "sufficient_information"


def test_graph_routes_directly_to_evaluator_when_no_candidate_exists() -> None:
    config = ResearchConfig()
    config.runtime.max_iterations = 10
    config.runtime.max_stagnation_cycles = 5
    config.runtime.max_consecutive_technical_failures = 2
    config.runtime.max_cycles_without_new_evidence = 5
    config.runtime.max_cycles_without_useful_sources = 5
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=InsufficientLLMWorkers(),
        browser=FailIfCalledBrowser(),
        search_client=EmptySearchClient(),
        telemetry=TelemetryRecorder(),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "What is happening to fusion demand?",
        max_iterations=10,
    )

    result = graph.invoke(initial_state)

    assert result["final_report"] is not None
    assert result["atomic_evidence"] == []
    assert result["final_report"].stop_reason == "research_exhausted"
    assert result["technical_reason"] in {"no_results", "no_queries"}
    assert result["consecutive_technical_failures"] >= 2


def test_graph_stops_when_final_synthesis_context_is_full() -> None:
    config = ResearchConfig()
    config.model.num_ctx = 400
    config.model.num_predict = 64
    config.runtime.synthesizer_prompt_margin = 0
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=FinalContextFullLLMWorkers(),
        browser=FakeBrowser(),
        search_client=FakeSearchClient(),
        telemetry=TelemetryRecorder(),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "What is happening to fusion demand?",
        max_iterations=4,
    )

    result = graph.invoke(initial_state)

    assert result["final_report"] is not None
    assert result["final_report"].stop_reason == "final_context_full"
    assert result["synthesis_budget"]["final_context_full"] is True


def test_graph_stops_when_research_is_exhausted_without_progress() -> None:
    config = ResearchConfig()
    config.runtime.max_iterations = 20
    config.runtime.max_stagnation_cycles = 2
    config.runtime.max_consecutive_technical_failures = 10
    config.runtime.max_cycles_without_new_evidence = 10
    config.runtime.max_cycles_without_useful_sources = 10
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=ExhaustedLLMWorkers(),
        browser=FakeBrowser(),
        search_client=FakeSearchClient(),
        telemetry=TelemetryRecorder(),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "What is happening to fusion demand?",
        max_iterations=20,
    )

    result = graph.invoke(initial_state)

    assert result["final_report"] is not None
    assert result["final_report"].stop_reason == "research_exhausted"
    assert result["stagnation_cycles"] >= 2
    assert result["cycles_without_new_evidence"] < 10


def test_graph_verbosity_zero_suppresses_telemetry_output() -> None:
    config = ResearchConfig()
    config.runtime.verbosity = 0
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=FakeLLMWorkers(),
        browser=FakeBrowser(),
        search_client=FakeSearchClient(),
        telemetry=TelemetryRecorder(verbosity=0),
    )
    graph = build_graph(runtime)

    result = graph.invoke(build_initial_state("What is happening to fusion demand?", max_iterations=4))

    assert result["telemetry"] == []


def test_graph_verbosity_three_includes_dossier_and_web_debug_events() -> None:
    config = ResearchConfig()
    config.runtime.verbosity = 3
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=FakeLLMWorkers(),
        browser=FakeBrowser(),
        search_client=FakeSearchClient(),
        telemetry=TelemetryRecorder(verbosity=3),
    )
    graph = build_graph(runtime)

    result = graph.invoke(build_initial_state("What is happening to fusion demand?", max_iterations=4))

    assert any(event.payload_type == "web_page" and "page" in event.payload for event in result["telemetry"])
    assert any(event.payload_type == "dossier_snapshot" for event in result["telemetry"])

