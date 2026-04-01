"""Planner node implementation."""

from __future__ import annotations
from typing import Any
from ..state import ResearchState
from .base import record_telemetry

class PlannerNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("planner", "Planning: {query}")
    def __call__(self, state: ResearchState) -> dict:
        payload, usage = self._runtime.llm_workers.plan_research_with_usage(self._runtime.context_manager.planner_context(state))
        
        known = {s.id for s in [*state["active_subqueries"], *state["resolved_subqueries"]]}
        new_sqs = [s for s in payload.subqueries if s.id not in known]
        llm_usage = {**state.get("llm_usage", {}), "planner": usage}
        
        event = self._runtime.telemetry.record("planner", "Agenda updated", new=len(new_sqs), **usage)
        return {
            "active_subqueries": [*state["active_subqueries"], *new_sqs],
            "search_intents": [*state["search_intents"], *payload.search_intents],
            "hypotheses": list(set([*state["hypotheses"], *payload.hypotheses])),
            "llm_usage": llm_usage,
            "telemetry": [*state["telemetry"], event]
        }
