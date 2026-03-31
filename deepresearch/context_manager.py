"""Hierarchical context manager.

Each node consumes a minimal projection of state. Selection is deterministic
first: active subqueries, actionable gaps, and the most relevant evidence. As
the dossier grows, the runtime can add semantic compression without changing
this contract.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .config import ResearchConfig
from .state import AtomicEvidence, Gap, ResearchState, Subquery
from .subagents.deterministic import estimate_tokens, select_evidence_for_context


class NodeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

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

    def _render_subqueries(self, subqueries: list[Subquery]) -> str:
        if not subqueries:
            return "- None"
        return "\n".join(
            f"- {item.id}: {item.question} | criteria={'; '.join(item.success_criteria[:3])}"
            for item in subqueries
        )

    def _render_gaps(self, gaps: list[Gap]) -> str:
        if not gaps:
            return "- None"
        return "\n".join(
            f"- {gap.subquery_id}: {gap.description} | suggested_queries={', '.join(gap.suggested_queries[:3])}"
            for gap in gaps[:5]
        )

    def _budget(self, ratio: float) -> int:
        available_tokens = max(0, self._config.model.num_ctx - self._config.model.num_predict)
        return int(available_tokens * ratio)

    def planner_context(self, state: ResearchState) -> NodeContext:
        has_subqueries = bool(state["active_subqueries"] or state["resolved_subqueries"])
        active = self._render_subqueries(state["active_subqueries"]) if has_subqueries else ""
        resolved = self._render_subqueries(state["resolved_subqueries"]) if has_subqueries else ""
        gaps = self._render_gaps(state["open_gaps"]) if has_subqueries else ""
        
        dossier_blocks = []
        if state["working_dossier"].global_summary:
            dossier_blocks.append(f"Global Summary: {state['working_dossier'].global_summary}")
        
        for sq_id, summary in state["working_dossier"].subquery_summaries.items():
            dossier_blocks.append(f"Subquery {sq_id} findings: {summary}")
        
        dossier_context = "\n".join(dossier_blocks)
        
        return NodeContext(
            query=state["query"],
            has_subqueries=has_subqueries,
            active_subqueries=active,
            resolved_subqueries=resolved,
            open_gaps=gaps,
            dossier_context=dossier_context,
        )

    def extractor_context(
        self,
        state: ResearchState,
        *,
        target_subquery_ids: list[str],
        local_source: str,
    ) -> NodeContext:
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=target_subquery_ids,
            budget_tokens=self._budget(self._config.context.evidence_budget_ratio),
        )
        return NodeContext(
            query=state["query"],
            active_subqueries=self._render_subqueries(state["active_subqueries"]),
            open_gaps=self._render_gaps(state["open_gaps"]),
            evidentiary=evidence,
            local_source=local_source,
        )

    def evaluator_context(self, state: ResearchState) -> NodeContext:
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=[item.id for item in state["active_subqueries"]],
            budget_tokens=self._budget(self._config.context.evidence_budget_ratio),
        )
        
        dossier_summary = ""
        if state["working_dossier"].global_summary:
            dossier_summary = f"Global Progress Summary: {state['working_dossier'].global_summary}\n\n"
        
        return NodeContext(
            query=state["query"],
            dossier_context=dossier_summary,
            active_subqueries=self._render_subqueries(state["active_subqueries"]),
            resolved_subqueries=self._render_subqueries(state["resolved_subqueries"]),
            open_gaps=self._render_gaps(state["open_gaps"]),
            evidentiary=evidence,
        )

    def synthesizer_context(self, state: ResearchState) -> NodeContext:
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=[item.id for item in state["resolved_subqueries"] or state["active_subqueries"]],
            budget_tokens=self._budget(self._config.context.evidence_budget_ratio),
        )
        budget = self._budget(self._config.context.dossier_budget_ratio)
        subquery_blocks = []
        for subquery in [*state["resolved_subqueries"], *state["active_subqueries"]]:
            summary = state["working_dossier"].subquery_summaries.get(subquery.id, "")
            if not summary:
                continue
            subquery_blocks.append(f"{subquery.id} | {subquery.question}\n{summary}")
        contradictions = "\n".join(
            f"- {item.topic}: {item.statement_a} <> {item.statement_b}"
            for item in state["contradictions"][:4]
        )
        gaps = self._render_gaps(state["open_gaps"])
        joined_subquery_blocks = "\n\n".join(subquery_blocks)
        dossier = "\n\n".join(
            block
            for block in (
                f"Global summary:\n{state['working_dossier'].global_summary}",
                f"Coverage by subquery:\n{joined_subquery_blocks}" if joined_subquery_blocks else "",
                f"Detected contradictions:\n{contradictions}" if contradictions else "",
                f"Open gaps:\n{gaps}" if gaps else "",
            )
            if block
        )
        if estimate_tokens(dossier) > budget:
            dossier = dossier[: budget * 4]
            
        return NodeContext(
            query=state["query"],
            dossier_context=dossier,
            evidentiary=evidence,
        )
