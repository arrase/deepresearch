"""Shared test fixtures and fake implementations for the research pipeline."""

from __future__ import annotations

import pytest

from deepresearch.config import ResearchConfig
from deepresearch.context_manager import ContextManager
from deepresearch.core.payloads import CoveragePayload, EvidenceDraft, EvidencePayload, PlannerPayload
from deepresearch.runtime import ResearchRuntime
from deepresearch.state import (
    AtomicEvidence,
    BrowserPageStatus,
    ConfidenceLevel,
    FinalReport,
    SearchCandidate,
    SearchIntent,
    SourceVisit,
    Subquery,
)
from deepresearch.telemetry import TelemetryRecorder

# ---------------------------------------------------------------------------
# Fake search clients
# ---------------------------------------------------------------------------

class FakeSearchClient:
    """Returns a single deterministic result for every query."""

    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                url="https://example.com/report",
                title="Example report",
                snippet=f"Result for {query}",
                domain="example.com",
            )
        ]


class EmptySearchClient:
    """Always returns zero results."""

    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        return []


class RecordingSearchClient:
    """Records every query it receives and returns a fixed candidate."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        self.calls.append(query)
        return [
            SearchCandidate(
                url="https://example.com/playwright-guide",
                title="Playwright guide",
                snippet="Playwright testing framework from Microsoft",
                domain="example.com",
            )
        ]


# ---------------------------------------------------------------------------
# Fake browsers
# ---------------------------------------------------------------------------

class FakeBrowser:
    """Returns a deterministic USEFUL page for every URL."""

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
    """Fails the test immediately if the browser is invoked."""

    def fetch(self, url: str) -> SourceVisit:
        raise AssertionError(f"Browser should not be called for url={url}")


# ---------------------------------------------------------------------------
# Fake LLM workers
# ---------------------------------------------------------------------------

class FakeLLMWorkers:
    """Deterministic LLM worker that always resolves sq_demo with a single evidence item."""

    def plan_research(self, context: object) -> PlannerPayload:
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

    def plan_research_with_usage(self, context: object) -> tuple[PlannerPayload, dict[str, int]]:
        return self.plan_research(context), {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}

    def extract_evidence(self, context: object) -> EvidencePayload:
        draft = EvidenceDraft(
            summary="Demand increased",
            claim="Fusion demand is rising in 2026.",
            quotation="Fusion demand is rising in 2026.",
            citation_locator="paragraph 1",
            relevance_score=0.9,
            confidence=ConfidenceLevel.HIGH,
        )
        return EvidencePayload(evidences=[draft])

    def extract_evidence_with_usage(self, context: object) -> tuple[EvidencePayload, dict[str, int]]:
        return self.extract_evidence(context), {"input_tokens": 20, "output_tokens": 8, "total_tokens": 28}

    def evaluate_coverage(self, context: object) -> CoveragePayload:
        return CoveragePayload(
            resolved_subquery_ids=["sq_demo"],
            is_sufficient=True,
            rationale="Enough evidence",
        )

    def evaluate_coverage_with_usage(self, context: object) -> tuple[CoveragePayload, dict[str, int]]:
        return self.evaluate_coverage(context), {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}

    def synthesize_report(self, context: object, query: str) -> FinalReport:
        return FinalReport(
            query=query,
            executive_answer="Fusion demand increased in 2026 according to the accepted source.",
            key_findings=["Demand increased in 2026"],
            confidence=ConfidenceLevel.HIGH,
            cited_sources=[{"url": "https://example.com/report", "title": "Example report", "evidence_ids": ["ev1"]}],
            evidence_ids=["ev1"],
            markdown_report="# Research Report\n\nFusion demand is rising.",
        )

    def synthesize_report_with_usage(self, context: object, query: str) -> tuple[FinalReport, dict[str, int]]:
        return self.synthesize_report(context, query), {"input_tokens": 30, "output_tokens": 40, "total_tokens": 70}

    def consume_telemetry_events(self) -> list:
        return []


class InsufficientLLMWorkers(FakeLLMWorkers):
    """LLM worker that never considers coverage sufficient."""

    def evaluate_coverage(self, context: object) -> CoveragePayload:
        return CoveragePayload(
            resolved_subquery_ids=[],
            is_sufficient=False,
            rationale="Need more evidence",
        )


class FinalContextFullLLMWorkers(InsufficientLLMWorkers):
    """Produces large evidence items so the synthesis context overflows quickly."""

    def extract_evidence(self, context: object) -> EvidencePayload:
        draft = EvidenceDraft(
            summary="Demand increased and costs remain high. " * 8,
            claim="Fusion demand is rising in 2026 while costs remain high.",
            quotation=("Fusion demand is rising in 2026 while costs remain high. " * 12).strip(),
            citation_locator="paragraph 1",
            relevance_score=0.9,
            confidence=ConfidenceLevel.HIGH,
        )
        return EvidencePayload(evidences=[draft])


class ExhaustedLLMWorkers(InsufficientLLMWorkers):
    """Produces zero evidence, causing research exhaustion."""

    def extract_evidence(self, context: object) -> EvidencePayload:
        return EvidencePayload(evidences=[])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def research_config() -> ResearchConfig:
    return ResearchConfig()


@pytest.fixture()
def context_manager(research_config: ResearchConfig) -> ContextManager:
    return ContextManager(research_config)


@pytest.fixture()
def fake_runtime(research_config: ResearchConfig, context_manager: ContextManager) -> ResearchRuntime:
    return ResearchRuntime(
        config=research_config,
        context_manager=context_manager,
        llm_workers=FakeLLMWorkers(),
        browser=FakeBrowser(),
        search_client=FakeSearchClient(),
        telemetry=TelemetryRecorder(),
    )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_subquery(**overrides: object) -> Subquery:
    """Create a Subquery with sensible defaults, overridable via kwargs."""
    defaults: dict[str, object] = {
        "question": "What is X?",
        "rationale": "Need definition",
        "evidence_target": 1,
        "search_terms": ["x definition"],
    }
    defaults.update(overrides)
    return Subquery(**defaults)  # type: ignore[arg-type]


def make_evidence(subquery_id: str = "sq_1", **overrides: object) -> AtomicEvidence:
    """Create an AtomicEvidence item with sensible defaults."""
    defaults: dict[str, object] = {
        "subquery_id": subquery_id,
        "source_url": "https://example.com",
        "source_title": "Example",
        "summary": "Summary of the claim",
        "claim": "X is a defined concept",
        "quotation": "X means ...",
        "citation_locator": "p1",
    }
    defaults.update(overrides)
    return AtomicEvidence(**defaults)  # type: ignore[arg-type]


def make_report(query: str = "Test query", markdown: str = "Short report") -> FinalReport:
    """Create a FinalReport with sensible defaults."""
    return FinalReport(
        query=query,
        executive_answer="Summary",
        key_findings=["Finding 1", "Finding 2"],
        confidence=ConfidenceLevel.HIGH,
        markdown_report=markdown,
        cited_sources=[],
    )
