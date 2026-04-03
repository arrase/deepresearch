"""Planner node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import summarize_subqueries
from ..state import ResearchState
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class PlannerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    @traceable(name="planner-node")
    @log_node_activity("planner", "Planning: {query}")
    def __call__(self, state: ResearchState) -> dict:
        payload, usage = self._runtime.llm_workers.plan_research_with_usage(
            self._runtime.context_manager.planner_context(state)
        )

        known = {s.id for s in [*state["active_subqueries"], *state["resolved_subqueries"]]}
        new_sqs = [s for s in payload.subqueries if s.id not in known]
        llm_usage = {**state.get("llm_usage", {}), "planner": usage}

        log_runtime_event(self._runtime, "[planner] Agenda updated", verbosity=1, new=len(new_sqs), **usage)
        log_runtime_event(
            self._runtime,
            "[planner] Planner decisions recorded",
            verbosity=2,
            new_subqueries=summarize_subqueries(new_sqs),
            search_intents=[intent.model_dump(mode="json") for intent in payload.search_intents[:5]],
            hypotheses=payload.hypotheses[:5],
        )
        return {
            "active_subqueries": [*state["active_subqueries"], *new_sqs],
            "search_intents": [*state["search_intents"], *payload.search_intents],
            "hypotheses": list(set([*state["hypotheses"], *payload.hypotheses])),
            "llm_usage": llm_usage,
        }
