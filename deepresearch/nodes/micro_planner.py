"""Micro-planner node: plans sub-topics for a single chapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..state import ResearchState, ResearchTopic, TopicStatus
from .base import log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class MicroPlannerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _find_chapter(self, state: ResearchState) -> ResearchTopic | None:
        """Return the current chapter, or pick the next incomplete one."""
        chapter_id = state.get("current_chapter_id")
        completed = set(state.get("completed_chapter_ids") or [])
        chapters = [t for t in state["plan"] if t.depth == 0]

        if chapter_id:
            return next((t for t in chapters if t.id == chapter_id), None)

        # Pick the next pending chapter not yet completed
        for ch in chapters:
            if ch.id not in completed and ch.status in {TopicStatus.PENDING, TopicStatus.IN_PROGRESS}:
                return ch
        return None

    @traceable(name="micro-planner-node")
    @log_node_activity("micro_planner", "Planning sub-topics: {query}")
    def __call__(self, state: ResearchState) -> dict:
        chapter = self._find_chapter(state)
        if chapter is None:
            log_runtime_event(self._runtime, "[micro_planner] No pending chapters", verbosity=1)
            return {"current_chapter_id": None}

        chapter_id = chapter.chapter_id or chapter.id
        context = self._runtime.context_manager.micro_planner_context(state, chapter)
        payload, usage = self._runtime.llm_workers.micro_plan_with_usage(context)

        max_depth = self._runtime.config.runtime.max_topic_depth
        known_ids = {t.id for t in state["plan"]}
        new_subtopics: list[ResearchTopic] = []
        for sub in payload.subtopics:
            if sub.id in known_ids:
                continue
            sub.parent_id = chapter_id
            sub.chapter_id = chapter_id
            sub.depth = min(sub.depth, max_depth) if sub.depth > 0 else 1
            if sub.depth > max_depth:
                continue
            sub.status = TopicStatus.PENDING
            new_subtopics.append(sub)

        plan = [*state["plan"], *new_subtopics]

        # Set active_topic_id to first pending topic of this chapter
        chapter_pending = [
            t for t in plan
            if t.chapter_id == chapter_id and t.status == TopicStatus.PENDING
        ]
        active_topic_id = chapter_pending[0].id if chapter_pending else state.get("active_topic_id")

        new_intents = list(payload.search_intents)
        # Tag intents with the chapter's topic ids
        for intent in new_intents:
            if not intent.topic_ids:
                intent.topic_ids = [sub.id for sub in new_subtopics[:1]] or [chapter_id]

        llm_usage = update_stage_llm_usage(state.get("llm_usage", {}), "micro_planner", usage)
        log_runtime_event(
            self._runtime,
            "[micro_planner] Sub-topics created",
            verbosity=1,
            chapter_id=chapter_id,
            new_subtopics=len(new_subtopics),
            **usage,
        )
        return {
            "plan": plan,
            "current_chapter_id": chapter_id,
            "active_topic_id": active_topic_id,
            "search_intents": [*state["search_intents"], *new_intents],
            "llm_usage": llm_usage,
        }
