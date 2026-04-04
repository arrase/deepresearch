"""Shared test fixtures and fake implementations for the SLM-oriented pipeline."""

from __future__ import annotations

import pytest

from deepresearch.config import ResearchConfig
from deepresearch.context_manager import ContextManager
from deepresearch.core.payloads import CoveragePayload, EvidenceDraft, EvidencePayload, PlannerPayload
from deepresearch.runtime import ResearchRuntime
from deepresearch.state import (
    BrowserPageStatus,
    ConfidenceLevel,
    CuratedEvidence,
    EvidenceSourceRef,
    FinalReport,
    Gap,
    ReportSource,
    ResearchTopic,
    SearchCandidate,
    SearchIntent,
    SourceVisit,
    TopicStatus,
)


class FakeSearchClient:
    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        return [
            SearchCandidate(
                url="https://example.com/report",
                normalized_url="https://example.com/report",
                title="Example report",
                snippet=f"Result for {query}",
                domain="example.com",
            )
        ]


class EmptySearchClient:
    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        return []


class RecordingSearchClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        self.calls.append(query)
        return [
            SearchCandidate(
                url="https://example.com/playwright-guide",
                normalized_url="https://example.com/playwright-guide",
                title="Playwright guide",
                snippet="Playwright testing framework from Microsoft",
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


class FakeLLMWorkers:
    def plan_research(self, context: object) -> PlannerPayload:
        topic = ResearchTopic(
            id="topic_demo",
            question="What happened to fusion demand in 2026?",
            rationale="Need primary claim",
            evidence_target=1,
            search_terms=["fusion demand 2026"],
            status=TopicStatus.PENDING,
        )
        return PlannerPayload(
            subqueries=[topic],
            search_intents=[SearchIntent(query="fusion demand 2026", rationale="primary", topic_ids=[topic.id])],
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
            resolved_subquery_ids=["topic_demo"],
            contradictions=[],
            open_gaps=[],
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
            reservations=[],
            open_gaps=[],
            cited_sources=[
                ReportSource(
                    url="https://example.com/report",
                    title="Example report",
                    evidence_ids=["evidence_demo"],
                )
            ],
            evidence_ids=["evidence_demo"],
            markdown_report="# Research Report\n\nFusion demand is rising.",
        )

    def synthesize_report_with_usage(self, context: object, query: str) -> tuple[FinalReport, dict[str, int]]:
        return self.synthesize_report(context, query), {"input_tokens": 30, "output_tokens": 40, "total_tokens": 70}


class InsufficientLLMWorkers(FakeLLMWorkers):
    def evaluate_coverage(self, context: object) -> CoveragePayload:
        return CoveragePayload(
            resolved_subquery_ids=[],
            contradictions=[],
            open_gaps=[Gap(topic_id="topic_demo", description="Need more evidence")],
            is_sufficient=False,
            rationale="Need more evidence",
        )


class FinalContextFullLLMWorkers(InsufficientLLMWorkers):
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
    def extract_evidence(self, context: object) -> EvidencePayload:
        return EvidencePayload(evidences=[])


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
    )


def make_topic(**overrides: object) -> ResearchTopic:
    defaults: dict[str, object] = {
        "question": "What is X?",
        "rationale": "Need definition",
        "evidence_target": 1,
        "search_terms": ["x definition"],
        "status": TopicStatus.PENDING,
    }
    defaults.update(overrides)
    return ResearchTopic.model_validate(defaults)


def make_curated_evidence(topic_id: str = "topic_1", **overrides: object) -> CuratedEvidence:
    defaults: dict[str, object] = {
        "topic_id": topic_id,
        "canonical_claim": "X is a defined concept",
        "summary": "Summary of the claim",
        "support_quotes": ["X means ..."],
        "sources": [EvidenceSourceRef(url="https://example.com", title="Example", locator="p1")],
        "prompt_fit_tokens_estimate": 30,
        "exact_generation_tokens": 10,
    }
    defaults.update(overrides)
    return CuratedEvidence.model_validate(defaults)
