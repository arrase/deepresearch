"""Meta-planner node: decomposes the research query into chapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..state import ResearchState, SearchIntent, TopicStatus
from .base import log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class MetaPlannerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    @traceable(name="meta-planner-node")
    @log_node_activity("meta_planner", "Decomposing query: {query}")
    def __call__(self, state: ResearchState) -> dict:
        context = self._runtime.context_manager.meta_planner_context(state)
        payload, usage = self._runtime.llm_workers.meta_plan_with_usage(context)

        max_chapters = self._runtime.config.runtime.max_chapters
        chapters = payload.chapters[:max_chapters]

        # Ensure every chapter is depth=0 with chapter_id=self.id
        for topic in chapters:
            topic.depth = 0
            topic.parent_id = None
            topic.chapter_id = topic.id
            topic.status = TopicStatus.PENDING

        # Inject a verbatim search intent for the first chapter
        first_ids = [chapters[0].id] if chapters else []
        verbatim = SearchIntent(
            query=state["query"].strip(),
            rationale="Verbatim user query",
            topic_ids=first_ids,
        )
        search_intents = [verbatim] if state["query"].strip() else []

        llm_usage = update_stage_llm_usage(state.get("llm_usage", {}), "meta_planner", usage)
        log_runtime_event(
            self._runtime,
            "[meta_planner] Chapters created",
            verbosity=1,
            chapters=len(chapters),
            **usage,
        )
        return {
            "plan": chapters,
            "hypotheses": list(dict.fromkeys([*state["hypotheses"], *payload.hypotheses])),
            "search_intents": [*state["search_intents"], *search_intents],
            "llm_usage": llm_usage,
        }
