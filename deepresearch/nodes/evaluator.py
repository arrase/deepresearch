"""Evaluator node implementation."""

from __future__ import annotations

from typing import Any

from ..core.utils import compute_minimum_coverage
from ..state import ResearchState
from .base import record_telemetry


class EvaluatorNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def _compute_progress_counters(
        self,
        state: ResearchState,
        *,
        newly_resolved_count: int,
        actionable_gap_count: int,
    ) -> dict[str, int]:
        runtime = self._runtime.config.runtime
        progress_score = state.get("progress_score", 0)
        progress_score += newly_resolved_count * runtime.weight_resolved_subquery
        progress_score += actionable_gap_count * runtime.weight_actionable_gap

        evidence_count = len(state.get("latest_evidence", []))
        useful_source_seen = bool(state.get("current_browser_result") and state["current_browser_result"].status.value in {"useful", "partial"})
        technical_reason = state.get("technical_reason")

        stagnation_cycles = 0 if progress_score >= runtime.min_progress_score_to_reset_stagnation else state["stagnation_cycles"] + 1
        consecutive_technical_failures = state["consecutive_technical_failures"] + 1 if technical_reason in {"no_results", "no_queries", "search_error"} else 0
        cycles_without_new_evidence = 0 if evidence_count > 0 else state["cycles_without_new_evidence"] + 1
        cycles_without_useful_sources = 0 if useful_source_seen else state["cycles_without_useful_sources"] + 1

        return {
            "progress_score": progress_score,
            "stagnation_cycles": stagnation_cycles,
            "consecutive_technical_failures": consecutive_technical_failures,
            "cycles_without_new_evidence": cycles_without_new_evidence,
            "cycles_without_useful_sources": cycles_without_useful_sources,
        }

    @record_telemetry("evaluator", "Evaluating: {query}")
    def __call__(self, state: ResearchState) -> dict:
        d_resolved_ids, d_gaps = compute_minimum_coverage(state["active_subqueries"], state["atomic_evidence"])
        semantic, usage = self._runtime.llm_workers.evaluate_coverage_with_usage(self._runtime.context_manager.evaluator_context(state))
        
        resolved_ids = set(d_resolved_ids) | {sid for sid in semantic.resolved_subquery_ids if any(e.subquery_id == sid for e in state["atomic_evidence"])}
        
        active, resolved = [], list(state["resolved_subqueries"])
        for sq in state["active_subqueries"]:
            if sq.id in resolved_ids:
                resolved.append(sq.model_copy(update={"status": "resolved"}))
            else:
                active.append(sq)

        open_gaps = list({(g.subquery_id, g.description): g for g in [*d_gaps, *semantic.open_gaps]}.values())
        
        synthesis_budget = self._runtime.context_manager.synthesis_budget(state)
        actionable_gap_count = sum(1 for gap in open_gaps if gap.actionable)
        newly_resolved_count = max(0, len(resolved) - len(state["resolved_subqueries"]))
        progress = self._compute_progress_counters(
            state,
            newly_resolved_count=newly_resolved_count,
            actionable_gap_count=actionable_gap_count,
        )
        stop_reason = state.get("stop_reason")
        if stop_reason is None and synthesis_budget.get("final_context_full"):
            stop_reason = "final_context_full"
        if stop_reason is None and (
            progress["stagnation_cycles"] >= self._runtime.config.runtime.max_stagnation_cycles
            or progress["consecutive_technical_failures"] >= self._runtime.config.runtime.max_consecutive_technical_failures
            or progress["cycles_without_new_evidence"] >= self._runtime.config.runtime.max_cycles_without_new_evidence
            or progress["cycles_without_useful_sources"] >= self._runtime.config.runtime.max_cycles_without_useful_sources
        ):
            stop_reason = "research_exhausted"
        if stop_reason is None and state["iteration"] >= state["max_iterations"]:
            stop_reason = "max_iterations_reached"
        is_sufficient = semantic.is_sufficient or stop_reason is not None
        llm_usage = {**state.get("llm_usage", {}), "evaluator": usage}

        event = self._runtime.telemetry.record(
            "evaluator",
            "Evaluated",
            resolved=len(resolved),
            active=len(active),
            sufficient=is_sufficient,
            stop_reason=stop_reason or ("sufficient_information" if semantic.is_sufficient else None),
            technical_reason=state.get("technical_reason"),
            progress_score=progress["progress_score"],
            stagnation_cycles=progress["stagnation_cycles"],
            consecutive_technical_failures=progress["consecutive_technical_failures"],
            cycles_without_new_evidence=progress["cycles_without_new_evidence"],
            cycles_without_useful_sources=progress["cycles_without_useful_sources"],
            iteration=state["iteration"],
            max_iterations=state["max_iterations"],
            **{k: v for k, v in synthesis_budget.items() if isinstance(v, (int, bool))},
            **usage,
        )
        return {
            "active_subqueries": active, "resolved_subqueries": resolved, "open_gaps": open_gaps,
            "contradictions": semantic.contradictions, "is_sufficient": is_sufficient,
            "stop_reason": stop_reason, "current_candidate": None, "llm_usage": llm_usage,
            "progress_score": progress["progress_score"],
            "stagnation_cycles": progress["stagnation_cycles"],
            "consecutive_technical_failures": progress["consecutive_technical_failures"],
            "cycles_without_new_evidence": progress["cycles_without_new_evidence"],
            "cycles_without_useful_sources": progress["cycles_without_useful_sources"],
            "synthesis_budget": synthesis_budget, "telemetry": [*state["telemetry"], event]
        }
