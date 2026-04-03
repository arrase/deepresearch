"""Research runtime dependencies and containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import ResearchConfig
    from .context_manager import ContextManager, NodeContext
    from .core.payloads import CoveragePayload, EvidencePayload, PlannerPayload
    from .state import FinalReport, SearchCandidate, SourceVisit


class SearchClientLike(Protocol):
    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]: ...


class BrowserLike(Protocol):
    def fetch(self, url: str) -> SourceVisit: ...


class LLMWorkersLike(Protocol):
    def plan_research(self, context: NodeContext) -> PlannerPayload: ...

    def plan_research_with_usage(self, context: NodeContext) -> tuple[PlannerPayload, dict[str, int]]: ...

    def extract_evidence(self, context: NodeContext) -> EvidencePayload: ...

    def extract_evidence_with_usage(self, context: NodeContext) -> tuple[EvidencePayload, dict[str, int]]: ...

    def evaluate_coverage(self, context: NodeContext) -> CoveragePayload: ...

    def evaluate_coverage_with_usage(self, context: NodeContext) -> tuple[CoveragePayload, dict[str, int]]: ...

    def synthesize_report(self, context: NodeContext, query: str) -> FinalReport: ...

    def synthesize_report_with_usage(
        self,
        context: NodeContext,
        query: str,
    ) -> tuple[FinalReport, dict[str, int]]: ...


@dataclass
class ResearchRuntime:
    config: ResearchConfig
    context_manager: ContextManager
    llm_workers: LLMWorkersLike
    browser: BrowserLike
    search_client: SearchClientLike
