"""Research runtime dependencies and containers."""

from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import ResearchConfig
    from .context_manager import ContextManager, NodeContext
    from .core.payloads import AuditPayload, EvidencePayload, MetaPlannerPayload, MicroPlannerPayload
    from .state import ChapterDraft, FinalReport, SearchCandidate


class SearchClientLike(Protocol):
    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]: ...


class LLMWorkersLike(Protocol):
    # -- meta-planner --
    def meta_plan(self, context: NodeContext) -> MetaPlannerPayload: ...

    def meta_plan_with_usage(self, context: NodeContext) -> tuple[MetaPlannerPayload, dict[str, int]]: ...

    # -- micro-planner --
    def micro_plan(self, context: NodeContext) -> MicroPlannerPayload: ...

    def micro_plan_with_usage(self, context: NodeContext) -> tuple[MicroPlannerPayload, dict[str, int]]: ...

    # -- evidence extraction --
    def extract_evidence(self, context: NodeContext) -> EvidencePayload: ...

    def extract_evidence_with_usage(self, context: NodeContext) -> tuple[EvidencePayload, dict[str, int]]: ...

    # -- auditor --
    def audit_evidence(self, context: NodeContext) -> AuditPayload: ...

    def audit_evidence_with_usage(self, context: NodeContext) -> tuple[AuditPayload, dict[str, int]]: ...

    # -- sub-synthesiser (per-chapter) --
    def sub_synthesize(self, context: NodeContext, chapter_id: str) -> ChapterDraft: ...

    def sub_synthesize_with_usage(
        self,
        context: NodeContext,
        chapter_id: str,
    ) -> tuple[ChapterDraft, dict[str, int]]: ...

    # -- global synthesiser (final report) --
    def global_synthesize(self, context: NodeContext, query: str) -> FinalReport: ...

    def global_synthesize_with_usage(
        self,
        context: NodeContext,
        query: str,
    ) -> tuple[FinalReport, dict[str, int]]: ...


@dataclass
class ResearchRuntime:
    config: ResearchConfig
    context_manager: ContextManager
    llm_workers: LLMWorkersLike
    search_client: SearchClientLike

    def close(self) -> None:
        for resource in (self.search_client, self.llm_workers):
            close_method = getattr(resource, "close", None)
            if callable(close_method):
                close_method()

    def __enter__(self) -> ResearchRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
