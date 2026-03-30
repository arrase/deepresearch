from deepresearch.config import ResearchConfig
from deepresearch.context_manager import ContextManager
from deepresearch.graph import build_graph
from deepresearch.nodes import ResearchRuntime
from deepresearch.state import (
    BrowserPageStatus,
    BrowserResult,
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
    def fetch(self, url: str) -> BrowserResult:
        return BrowserResult(
            url=url,
            final_url=url,
            status=BrowserPageStatus.USEFUL,
            title="Example report",
            content="# Example report\nFusion demand is rising in 2026.",
            excerpt="Fusion demand is rising in 2026.",
            exit_code=0,
        )


class FailIfCalledBrowser:
    def fetch(self, url: str) -> BrowserResult:
        raise AssertionError(f"Browser should not be called for url={url}")


class EmptySearchClient:
    def search(self, query: str, *, max_results: int | None = None):
        return []


class FakeLLMWorkers:
    def plan_research(self, context):
        subquery = Subquery(
            id="sq_demo",
            question="What happened to fusion demand in 2026?",
            rationale="Need primary claim",
            evidence_target=1,
            search_terms=["fusion demand 2026"],
        )
        return type(
            "PlannerPayload",
            (),
            {
                "subqueries": [subquery],
                "search_intents": [SearchIntent(query="fusion demand 2026", rationale="primary", subquery_ids=[subquery.id])],
                "hypotheses": ["Demand increased"],
            },
        )()

    def extract_evidence(self, context):
        draft = type(
            "EvidenceDraft",
            (),
            {
                "summary": "Demand increased",
                "claim": "Fusion demand is rising in 2026.",
                "quotation": "Fusion demand is rising in 2026.",
                "citation_locator": "paragraph 1",
                "relevance_score": 0.9,
                "confidence": ConfidenceLevel.HIGH,
                "caveats": [],
                "tags": ["trend"],
            },
        )()
        return type("EvidencePayload", (), {"evidences": [draft]})()

    def evaluate_coverage(self, context):
        return type(
            "CoveragePayload",
            (),
            {
                "resolved_subquery_ids": ["sq_demo"],
                "contradictions": [],
                "open_gaps": [],
                "is_sufficient": True,
                "rationale": "Enough evidence",
            },
        )()

    def synthesize_report(self, context, *, query: str):
        return FinalReport(
            query=query,
            executive_answer="Fusion demand increased in 2026 according to the accepted source.",
            key_findings=["Demand increased in 2026"],
            confidence=ConfidenceLevel.HIGH,
            reservations=[],
            open_gaps=[],
            cited_sources=[{"url": "https://example.com/report", "title": "Example report", "evidence_ids": ["ev1"]}],
            evidence_ids=["ev1"],
        )


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
        target_tokens=100000,
        configured_by="test",
        selection_policy="hierarchical_relevance_first",
    )
    result = graph.invoke(initial_state)
    assert result["final_report"] is not None
    assert result["final_report"].executive_answer.startswith("Fusion demand increased")
    assert result["final_report"].markdown_report.startswith("# Research Report")
    assert result["atomic_evidence"]


def test_graph_routes_directly_to_evaluator_when_no_candidate_exists() -> None:
    config = ResearchConfig()
    config.runtime.max_iterations = 1
    runtime = ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=FakeLLMWorkers(),
        browser=FailIfCalledBrowser(),
        search_client=EmptySearchClient(),
        telemetry=TelemetryRecorder(),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "What is happening to fusion demand?",
        max_iterations=1,
        target_tokens=100000,
        configured_by="test",
        selection_policy="hierarchical_relevance_first",
    )

    result = graph.invoke(initial_state)

    assert result["final_report"] is not None
    assert result["atomic_evidence"] == []
