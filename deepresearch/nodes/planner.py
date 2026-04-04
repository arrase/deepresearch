"""Planner node implementation for the SLM-oriented pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import summarize_subqueries
from ..state import ResearchState, TopicStatus
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

        known_ids = {topic.id for topic in state["plan"]}
        new_topics = [topic for topic in payload.subqueries if topic.id not in known_ids]
        plan = [*state["plan"], *new_topics]
        active_topic_id = state["active_topic_id"]
        if active_topic_id is None:
            pending_topics = sorted(
                [topic for topic in plan if topic.status == TopicStatus.PENDING],
                key=lambda topic: (topic.priority, topic.id),
            )
            if pending_topics:
                active_topic_id = pending_topics[0].id

        llm_usage = {**state.get("llm_usage", {}), "planner": usage}
        log_runtime_event(self._runtime, "[planner] Agenda updated", verbosity=1, new=len(new_topics), **usage)
        log_runtime_event(
            self._runtime,
            "[planner] Planner decisions recorded",
            verbosity=2,
            new_topics=summarize_subqueries(new_topics),
            search_intents=[intent.model_dump(mode="json") for intent in payload.search_intents[:5]],
            hypotheses=payload.hypotheses[:5],
        )
        return {
            "plan": plan,
            "active_topic_id": active_topic_id,
            "search_intents": [*state["search_intents"], *payload.search_intents],
            "hypotheses": list(dict.fromkeys([*state["hypotheses"], *payload.hypotheses])),
            "llm_usage": llm_usage,
            "replan_requested": False,
            "technical_reason": None,
        }
