from deepresearch.config import ResearchConfig
from deepresearch.context_manager import ContextManager
from deepresearch.core.payloads import CoveragePayload, EvidenceDraft, EvidencePayload, PlannerPayload
from deepresearch.graph import build_graph
from deepresearch.nodes.evaluator import EvaluatorNode
from deepresearch.nodes.extractor import ExtractorNode
from deepresearch.nodes.source_manager import SourceManagerNode
from deepresearch.runtime import ResearchRuntime
from deepresearch.state import (
    BrowserPageStatus,
    ConfidenceLevel,
    FinalReport,
    Gap,
    ReportSource,
    SearchCandidate,
    SearchIntent,
    SourceVisit,
    Subquery,
    build_initial_state,
)


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
        )


class FailIfCalledBrowser:
    def fetch(self, url: str) -> SourceVisit:
        raise AssertionError(f"Browser should not be called for url={url}")


class EmptySearchClient:
    def search(self, query: str, *, max_results: int | None = None):
        return []


class RecordingSearchClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int | None = None):
        self.calls.append(query)
        return [
            SearchCandidate(
                url="https://example.com/playwright-guide",
                title="Playwright guide",
                snippet="Playwright testing framework from Microsoft",
                domain="example.com",
            )
        ]


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
            cited_sources=[
                ReportSource(
                    url="https://example.com/report",
                    title="Example report",
                    evidence_ids=["ev1"],
                )
            ],
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


def test_source_manager_prioritizes_gap_queries_over_existing_queue() -> None:
    config = ResearchConfig()
    search_client = RecordingSearchClient()
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=FakeLLMWorkers(),
        browser=FakeBrowser(),
        search_client=search_client,
    )
    node = SourceManagerNode(runtime)
    state = build_initial_state("Lightpanda vs Playwright", max_iterations=4)
    state["active_subqueries"] = [
        Subquery(id="sq_1", question="What is Lightpanda?", rationale="r", search_terms=["lightpanda"]),
        Subquery(id="sq_2", question="What is Playwright?", rationale="r", search_terms=["playwright"]),
    ]
    state["search_queue"] = [
        SearchCandidate(
            url="https://example.com/old",
            title="Old queue item",
            domain="example.com",
            subquery_ids=["sq_1"],
        )
    ]
    state["open_gaps"] = [
        Gap(subquery_id="sq_2", description="Need Playwright docs", suggested_queries=["playwright framework docs"])
    ]

    result = node(state)

    assert search_client.calls == ["playwright framework docs"]
    assert result["current_candidate"].url == "https://example.com/playwright-guide"
    assert result["current_candidate"].subquery_ids == ["sq_2"]
    assert any(candidate.url == "https://example.com/old" for candidate in result["search_queue"])


def test_extractor_assigns_evidence_to_candidate_subquery_not_first_active() -> None:
    config = ResearchConfig()
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=FakeLLMWorkers(),
        browser=FakeBrowser(),
        search_client=FakeSearchClient(),
    )
    node = ExtractorNode(runtime)
    state = build_initial_state("Lightpanda vs Playwright", max_iterations=4)
    state["active_subqueries"] = [
        Subquery(id="sq_1", question="What is Lightpanda?", rationale="r", search_terms=["lightpanda framework"]),
        Subquery(
            id="sq_2",
            question="What is Playwright?",
            rationale="r",
            search_terms=["playwright testing framework"],
        ),
    ]
    state["current_candidate"] = SearchCandidate(
        url="https://example.com/playwright-guide",
        title="Playwright guide",
        snippet="Playwright testing framework from Microsoft",
        domain="example.com",
        subquery_ids=["sq_2"],
    )
    state["current_browser_result"] = SourceVisit(
        url="https://example.com/playwright-guide",
        final_url="https://example.com/playwright-guide",
        status=BrowserPageStatus.USEFUL,
        title="Playwright guide",
        content="Playwright is a testing framework maintained by Microsoft.",
        excerpt="Playwright is a testing framework maintained by Microsoft.",
    )

    result = node(state)

    assert result["latest_evidence"]
    assert result["latest_evidence"][0].subquery_id == "sq_2"


def test_evaluator_counts_zero_evidence_cycle_as_stagnation() -> None:
    config = ResearchConfig()
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=FakeLLMWorkers(),
        browser=FakeBrowser(),
        search_client=FakeSearchClient(),
    )
    node = EvaluatorNode(runtime)
    state = build_initial_state("Lightpanda vs Playwright", max_iterations=4)
    state["progress_score"] = config.runtime.weight_useful_source
    state["stagnation_cycles"] = 1
    state["latest_evidence"] = []

    progress = node._compute_progress_counters(state, newly_resolved_count=0)

    assert progress["progress_score"] == config.runtime.weight_useful_source
    assert progress["stagnation_cycles"] == 2
