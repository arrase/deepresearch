"""Evaluator node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import compute_minimum_coverage, enrich_gaps_with_search_terms, summarize_gaps
from ..state import ResearchState
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class EvaluatorNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _compute_progress_counters(
        self,
        state: ResearchState,
        *,
        newly_resolved_count: int,
    ) -> dict[str, int]:
        runtime = self._runtime.config.runtime
        progress_score = state.get("progress_score", 0)
        progress_score += newly_resolved_count * runtime.weight_resolved_subquery

        evidence_count = len(state.get("latest_evidence", []))
        current_browser_result = state.get("current_browser_result")
        useful_source_seen = bool(
            current_browser_result
            and current_browser_result.status.value in {"useful", "partial"}
        )
        technical_reason = state.get("technical_reason")

        stagnation_cycles = (
            0
            if progress_score >= runtime.min_progress_score_to_reset_stagnation
            else state["stagnation_cycles"] + 1
        )
        consecutive_technical_failures = (
            state["consecutive_technical_failures"] + 1
            if technical_reason in {"no_results", "no_queries", "search_error"}
            else 0
        )
        cycles_without_new_evidence = 0 if evidence_count > 0 else state["cycles_without_new_evidence"] + 1
        cycles_without_useful_sources = 0 if useful_source_seen else state["cycles_without_useful_sources"] + 1

        return {
            "progress_score": progress_score,
            "stagnation_cycles": stagnation_cycles,
            "consecutive_technical_failures": consecutive_technical_failures,
            "cycles_without_new_evidence": cycles_without_new_evidence,
            "cycles_without_useful_sources": cycles_without_useful_sources,
        }

    @traceable(name="evaluator-node")
    @log_node_activity("evaluator", "Evaluating: {query}")
    def __call__(self, state: ResearchState) -> dict:
        d_resolved_ids, d_gaps = compute_minimum_coverage(state["active_subqueries"], state["atomic_evidence"])
        semantic, usage = self._runtime.llm_workers.evaluate_coverage_with_usage(
            self._runtime.context_manager.evaluator_context(state)
        )

        resolved_ids = set(d_resolved_ids) | {
            sid
            for sid in semantic.resolved_subquery_ids
            if any(e.subquery_id == sid for e in state["atomic_evidence"])
        }

        active, resolved = [], list(state["resolved_subqueries"])
        for sq in state["active_subqueries"]:
            if sq.id in resolved_ids:
                resolved.append(sq.model_copy(update={"status": "resolved"}))
            else:
                active.append(sq)

        open_gaps = list({(g.subquery_id, g.description): g for g in [*d_gaps, *semantic.open_gaps]}.values())
        open_gaps = enrich_gaps_with_search_terms(open_gaps, active)

        synthesis_budget = self._runtime.context_manager.synthesis_budget(state)
        newly_resolved_count = max(0, len(resolved) - len(state["resolved_subqueries"]))
        progress = self._compute_progress_counters(
            state,
            newly_resolved_count=newly_resolved_count,
        )
        stop_reason = state.get("stop_reason")
        if stop_reason is None and synthesis_budget.get("final_context_full"):
            stop_reason = "final_context_full"
        if stop_reason is None and (
            progress["stagnation_cycles"] >= self._runtime.config.runtime.max_stagnation_cycles
            or progress["consecutive_technical_failures"]
            >= self._runtime.config.runtime.max_consecutive_technical_failures
            or progress["cycles_without_new_evidence"] >= self._runtime.config.runtime.max_cycles_without_new_evidence
            or progress["cycles_without_useful_sources"]
            >= self._runtime.config.runtime.max_cycles_without_useful_sources
        ):
            stop_reason = "research_exhausted"
        if stop_reason is None and state["iteration"] >= state["max_iterations"]:
            stop_reason = "max_iterations_reached"
        is_sufficient = semantic.is_sufficient or stop_reason is not None
        llm_usage = {**state.get("llm_usage", {}), "evaluator": usage}

        log_runtime_event(
            self._runtime,
            "[evaluator] Evaluated",
            verbosity=1,
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
        log_runtime_event(
            self._runtime,
            "[evaluator] Coverage decision details",
            verbosity=2,
            deterministic_resolved_ids=d_resolved_ids,
            semantic_resolved_ids=semantic.resolved_subquery_ids,
            combined_resolved_ids=sorted(resolved_ids),
            semantic_rationale=semantic.rationale,
            open_gaps=summarize_gaps(open_gaps),
            contradictions=[item.model_dump(mode="json") for item in semantic.contradictions[:5]],
        )
        log_runtime_event(
            self._runtime,
            "[evaluator] Coverage snapshot after evaluation",
            verbosity=3,
            snapshot=self._runtime.context_manager.debug_state_snapshot({
                **state,
                "active_subqueries": active,
                "resolved_subqueries": resolved,
                "open_gaps": open_gaps,
            }),
        )
        return {
            "active_subqueries": active,
            "resolved_subqueries": resolved,
            "open_gaps": open_gaps,
            "contradictions": semantic.contradictions,
            "is_sufficient": is_sufficient,
            "stop_reason": stop_reason,
            "current_candidate": None,
            "llm_usage": llm_usage,
            "urls_visited_since_eval": 0,
            "progress_score": progress["progress_score"],
            "stagnation_cycles": progress["stagnation_cycles"],
            "consecutive_technical_failures": progress["consecutive_technical_failures"],
            "cycles_without_new_evidence": progress["cycles_without_new_evidence"],
            "cycles_without_useful_sources": progress["cycles_without_useful_sources"],
            "synthesis_budget": synthesis_budget,
        }
