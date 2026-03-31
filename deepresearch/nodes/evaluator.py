"""Evaluator node implementation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..core.utils import compute_minimum_coverage
from ..state import Gap, GapSeverity, ResearchState, Subquery
from .base import record_telemetry


class EvaluatorNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("evaluator", "Evaluating research coverage for: {query}")
    def __call__(self, state: ResearchState) -> dict:
        deterministic_resolved_ids, deterministic_gaps = compute_minimum_coverage(
            active_subqueries=state["active_subqueries"],
            evidence=state["atomic_evidence"],
        )
        context = self._runtime.context_manager.evaluator_context(state)
        semantic = self._runtime.llm_workers.evaluate_coverage(context)
        resolved_ids = set(deterministic_resolved_ids)
        evidence_count_by_subquery = Counter(item.subquery_id for item in state["atomic_evidence"])
        for subquery_id in semantic.resolved_subquery_ids:
            if evidence_count_by_subquery[subquery_id] >= 1:
                resolved_ids.add(subquery_id)

        remaining_active: list[Subquery] = []
        newly_resolved: list[Subquery] = []
        for subquery in state["active_subqueries"]:
            if subquery.id in resolved_ids:
                newly_resolved.append(subquery.model_copy(update={"status": "resolved"}))
            else:
                remaining_active.append(subquery)

        gap_index: dict[tuple[str, str], Gap] = {}
        for gap in [*deterministic_gaps, *semantic.open_gaps]:
            gap_index[(gap.subquery_id, gap.description)] = gap
        open_gaps = list(gap_index.values())

        # Dynamic evidence target adjustment
        contradiction_subquery_ids = set()
        for contradiction in semantic.contradictions:
            for ev_id in contradiction.evidence_ids:
                for ev in state["atomic_evidence"]:
                    if ev.id == ev_id:
                        contradiction_subquery_ids.add(ev.subquery_id)
                        break
        
        gap_subquery_ids = {gap.subquery_id for gap in open_gaps if gap.severity in {GapSeverity.HIGH, GapSeverity.CRITICAL}}
        
        adjusted_active = []
        for subquery in remaining_active:
            new_target = subquery.evidence_target
            if subquery.id in contradiction_subquery_ids or subquery.id in gap_subquery_ids:
                new_target = min(subquery.evidence_target + 2, 10)
                self._runtime.telemetry.record(
                    "evaluator",
                    "Increasing evidence target due to uncertainty or contradiction",
                    subquery_id=subquery.id,
                    new_target=new_target,
                )
            adjusted_active.append(subquery.model_copy(update={"evidence_target": new_target}))
        remaining_active = adjusted_active

        fallback_reason = state["fallback_reason"]
        if state["iteration"] >= state["max_iterations"] and fallback_reason is None:
            fallback_reason = "max_iterations_reached"

        is_sufficient = False
        resolved_total = len(state["resolved_subqueries"]) + len(newly_resolved)
        minimum_report_evidence = 1 if resolved_total <= 1 else max(2, resolved_total)
        
        # Stop criteria logic
        if semantic.is_sufficient and not remaining_active and len(state["atomic_evidence"]) >= minimum_report_evidence:
            is_sufficient = True
        elif not remaining_active and not open_gaps and len(state["atomic_evidence"]) >= minimum_report_evidence:
            is_sufficient = True
        elif fallback_reason is not None and len(state["atomic_evidence"]) >= 1:
            is_sufficient = True
        elif fallback_reason in ("search_backend_failure", "no_actionable_sources") and state["iteration"] >= 2:
            is_sufficient = True
        elif state["iteration"] >= state["max_iterations"]:
            is_sufficient = True

        event = self._runtime.telemetry.record(
            "evaluator",
            "Coverage evaluated",
            resolved=len(newly_resolved),
            remaining=len(remaining_active),
            contradictions=len(semantic.contradictions),
            fallback_reason=fallback_reason,
            is_sufficient=is_sufficient,
        )
        return {
            "active_subqueries": remaining_active,
            "resolved_subqueries": [*state["resolved_subqueries"], *newly_resolved],
            "open_gaps": open_gaps,
            "contradictions": semantic.contradictions,
            "is_sufficient": is_sufficient,
            "fallback_reason": fallback_reason,
            "current_candidate": None,
            "telemetry": [*state["telemetry"], event],
        }
