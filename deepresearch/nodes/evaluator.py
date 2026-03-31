"""Evaluator node implementation."""

from __future__ import annotations

from typing import Any

from ..core.utils import compute_minimum_coverage
from ..state import ResearchState
from .base import record_telemetry


class EvaluatorNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("evaluator", "Evaluating: {query}")
    def __call__(self, state: ResearchState) -> dict:
        d_resolved_ids, d_gaps = compute_minimum_coverage(state["active_subqueries"], state["atomic_evidence"])
        semantic = self._runtime.llm_workers.evaluate_coverage(self._runtime.context_manager.evaluator_context(state))
        
        resolved_ids = set(d_resolved_ids) | {sid for sid in semantic.resolved_subquery_ids if any(e.subquery_id == sid for e in state["atomic_evidence"])}
        
        active, resolved = [], list(state["resolved_subqueries"])
        for sq in state["active_subqueries"]:
            if sq.id in resolved_ids:
                resolved.append(sq.model_copy(update={"status": "resolved"}))
            else:
                active.append(sq)

        open_gaps = list({(g.subquery_id, g.description): g for g in [*d_gaps, *semantic.open_gaps]}.values())
        
        fallback = state["fallback_reason"] or ("max_iterations_reached" if state["iteration"] >= state["max_iterations"] else None)
        is_sufficient = semantic.is_sufficient or not active or fallback is not None

        event = self._runtime.telemetry.record("evaluator", "Evaluated", resolved=len(resolved), active=len(active), sufficient=is_sufficient)
        return {
            "active_subqueries": active, "resolved_subqueries": resolved, "open_gaps": open_gaps,
            "contradictions": semantic.contradictions, "is_sufficient": is_sufficient,
            "fallback_reason": fallback, "current_candidate": None, "telemetry": [*state["telemetry"], event]
        }
