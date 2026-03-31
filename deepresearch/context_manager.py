"""Hierarchical context manager."""

from __future__ import annotations
from pydantic import BaseModel, Field
from .config import ResearchConfig
from .state import AtomicEvidence, Gap, ResearchState, Subquery
from .core.utils import estimate_tokens, select_evidence_for_context

class NodeContext(BaseModel):
    query: str
    has_subqueries: bool = False
    active_subqueries: str = ""
    resolved_subqueries: str = ""
    open_gaps: str = ""
    dossier_context: str = ""
    evidentiary: list[AtomicEvidence] = Field(default_factory=list)
    local_source: str = ""

class ContextManager:
    def __init__(self, config: ResearchConfig) -> None:
        self._config = config

    def _render_sq(self, sqs: list[Subquery]) -> str:
        return "\n".join(f"- {s.id}: {s.question}" for s in sqs) or "- None"

    def _render_gaps(self, gaps: list[Gap]) -> str:
        return "\n".join(f"- {g.subquery_id}: {g.description}" for g in gaps[:5]) or "- None"

    def _budget(self, ratio: float) -> int:
        return int(max(0, self._config.model.num_ctx - self._config.model.num_predict) * ratio)

    def planner_context(self, state: ResearchState) -> NodeContext:
        has = bool(state["active_subqueries"] or state["resolved_subqueries"])
        dossier = "\n".join(f"Subquery {sid}: {sum}" for sid, sum in state["working_dossier"].subquery_summaries.items())
        return NodeContext(
            query=state["query"], has_subqueries=has,
            active_subqueries=self._render_sq(state["active_subqueries"]),
            resolved_subqueries=self._render_sq(state["resolved_subqueries"]),
            open_gaps=self._render_gaps(state["open_gaps"]),
            dossier_context=dossier
        )

    def extractor_context(self, state: ResearchState, targets: list[str], local_source: str) -> NodeContext:
        evidence = select_evidence_for_context(state["atomic_evidence"], subquery_ids=targets, budget_tokens=self._budget(self._config.context.evidence_budget_ratio))
        return NodeContext(query=state["query"], active_subqueries=self._render_sq(state["active_subqueries"]), open_gaps=self._render_gaps(state["open_gaps"]), evidentiary=evidence, local_source=local_source)

    def evaluator_context(self, state: ResearchState) -> NodeContext:
        evidence = select_evidence_for_context(state["atomic_evidence"], subquery_ids=[s.id for s in state["active_subqueries"]], budget_tokens=self._budget(self._config.context.evidence_budget_ratio))
        return NodeContext(query=state["query"], active_subqueries=self._render_sq(state["active_subqueries"]), resolved_subqueries=self._render_sq(state["resolved_subqueries"]), open_gaps=self._render_gaps(state["open_gaps"]), evidentiary=evidence)

    def synthesizer_context(self, state: ResearchState) -> NodeContext:
        evidence = select_evidence_for_context(state["atomic_evidence"], subquery_ids=[s.id for s in state["resolved_subqueries"] or state["active_subqueries"]], budget_tokens=self._budget(self._config.context.evidence_budget_ratio))
        dossier = "\n\n".join(f"{s.id} | {s.question}\n{state['working_dossier'].subquery_summaries.get(s.id, '')}" for s in [*state["resolved_subqueries"], *state["active_subqueries"]])
        if estimate_tokens(dossier) > self._budget(self._config.context.dossier_budget_ratio):
            dossier = dossier[:self._budget(self._config.context.dossier_budget_ratio)*4]
        return NodeContext(query=state["query"], dossier_context=dossier, evidentiary=evidence)
