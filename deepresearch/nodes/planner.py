"""Planner node implementation."""

from __future__ import annotations

from typing import Any

from ..state import ResearchState
from .base import record_telemetry


class PlannerNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("planner", "Starting planning for query: {query}")
    def __call__(self, state: ResearchState) -> dict:
        context = self._runtime.context_manager.planner_context(state)
        payload = self._runtime.llm_workers.plan_research(context)
        
        # Deduplicate and merge subqueries
        existing_ids = {sq.id for sq in [*state["active_subqueries"], *state["resolved_subqueries"]]}
        merged_subqueries = list(state["active_subqueries"])
        added_count = 0
        for nsq in payload.subqueries:
            if nsq.id not in existing_ids:
                merged_subqueries.append(nsq)
                added_count += 1
        
        # Merge search intents and hypotheses
        merged_intents = [*state["search_intents"], *payload.search_intents]
        merged_hypotheses = list(set([*state["hypotheses"], *payload.hypotheses]))

        event = self._runtime.telemetry.record(
            "planner",
            "Research agenda updated",
            added_subqueries=added_count,
            total_active=len(merged_subqueries),
            search_intents=len(merged_intents),
        )
        return {
            "active_subqueries": merged_subqueries,
            "search_intents": merged_intents,
            "hypotheses": merged_hypotheses,
            "telemetry": [*state["telemetry"], event],
        }
