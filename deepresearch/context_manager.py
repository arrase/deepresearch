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

    permanent: str
    strategic: str
    operational: str
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
        return int(self._config.context.target_tokens * ratio)

    def planner_context(self, state: ResearchState) -> NodeContext:
        return NodeContext(
            permanent=f"Primary question: {state['query']}",
            strategic="No subqueries exist yet. You must produce the initial agenda.",
            operational=(
                "Decompose the question into concrete subqueries, working hypotheses, and initial search queries."
            ),
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
            permanent=f"Primary question: {state['query']}",
            strategic=(
                f"Active subqueries:\n{self._render_subqueries(state['active_subqueries'])}\n"
                f"Open gaps:\n{self._render_gaps(state['open_gaps'])}"
            ).strip(),
            operational="Extract traceable atomic evidence without inferring beyond the source text.",
            evidentiary=evidence,
            local_source=local_source,
        )

    def evaluator_context(self, state: ResearchState) -> NodeContext:
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=[item.id for item in state["active_subqueries"]],
            budget_tokens=self._budget(self._config.context.evidence_budget_ratio),
        )
        strategic = (
            f"Active subqueries:\n{self._render_subqueries(state['active_subqueries'])}\n"
            f"Resolved subqueries:\n{self._render_subqueries(state['resolved_subqueries'])}\n"
            f"Open gaps:\n{self._render_gaps(state['open_gaps'])}"
        ).strip()
        operational = (
            "Evaluate coverage, contradictions, and sufficiency. Do not stop the research if the evidence does not support the answer."
        )
        return NodeContext(
            permanent=f"Primary question: {state['query']}",
            strategic=strategic,
            operational=operational,
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
            permanent=f"Primary question: {state['query']}",
            strategic=dossier,
            operational=(
                "Write a final conceptual Markdown report with an executive summary, thematic analysis, confidence, reservations, gaps, and citations. "
                "Every block must rely only on relevant evidence."
            ),
            evidentiary=evidence,
        )
