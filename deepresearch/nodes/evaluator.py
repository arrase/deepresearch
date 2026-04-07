"""Evaluator node: purely deterministic coverage assessment per chapter.

The evaluator no longer invokes the LLM.  It counts evidence, tracks
stagnation, and decides whether the current chapter is ready for audit or
needs more source discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import compute_minimum_coverage, enrich_gaps_with_search_terms, summarize_gaps
from ..state import ResearchState, ResearchTopic, StopReason, SynthesisBudget, TopicStatus
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class EvaluatorNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    # ------------------------------------------------------------------
    # Stop-reason logic
    # ------------------------------------------------------------------

    def _determine_stop_reason(
        self,
        *,
        plan: list[ResearchTopic],
        synthesis_budget: SynthesisBudget,
        current_iteration: int,
        max_iterations: int,
        cycles_without_new_evidence: int,
        cycles_without_useful_sources: int,
        consecutive_technical_failures: int,
    ) -> str | None:
        if synthesis_budget.final_context_full:
            return StopReason.CONTEXT_SATURATION.value
        if current_iteration >= max_iterations:
            return StopReason.MAX_ITERATIONS_REACHED.value
        runtime = self._runtime.config.runtime
        stuck_by_evidence = cycles_without_new_evidence >= runtime.max_cycles_without_new_evidence
        stuck_by_sources = cycles_without_useful_sources >= runtime.max_cycles_without_useful_sources
        stuck_by_technical = consecutive_technical_failures >= runtime.max_consecutive_technical_failures
        if stuck_by_technical or (stuck_by_evidence and stuck_by_sources):
            return StopReason.STUCK_NO_SOURCES.value
        return None

    # ------------------------------------------------------------------
    # Topic status updates (chapter-scoped)
    # ------------------------------------------------------------------

    def _update_topic_statuses(
        self,
        state: ResearchState,
        deterministic_resolved: list[str],
    ) -> list[ResearchTopic]:
        chapter_id = state.get("current_chapter_id") or ""
        coverage = state["topic_coverage"]
        min_attempts = self._runtime.config.runtime.min_attempts_before_exhaustion
        active_topic_id = state.get("active_topic_id")
        plan: list[ResearchTopic] = []

        for topic in state["plan"]:
            # Only touch topics of the current chapter
            if chapter_id and topic.chapter_id != chapter_id:
                plan.append(topic)
                continue

            topic_coverage = coverage.get(topic.id)
            accepted_count = topic_coverage.accepted_evidence_count if topic_coverage else 0
            attempts = state["topic_attempts"].get(topic.id, 0)
            completed = topic.id in deterministic_resolved

            exhausted = (
                topic.id == active_topic_id
                and accepted_count == 0
                and attempts >= min_attempts
                and state.get("technical_reason") in {"no_results", "no_topics", "search_error"}
            )

            if completed:
                plan.append(topic.model_copy(update={"status": TopicStatus.COMPLETED}))
            elif exhausted:
                plan.append(topic.model_copy(update={"status": TopicStatus.EXHAUSTED}))
            elif topic.id == active_topic_id:
                plan.append(topic.model_copy(update={"status": TopicStatus.IN_PROGRESS}))
            else:
                plan.append(topic)
        return plan

    # ------------------------------------------------------------------
    # Stagnation counters
    # ------------------------------------------------------------------

    def _counter_updates(self, state: ResearchState) -> dict[str, int]:
        return {
            "cycles_without_new_evidence": (
                0 if state["new_evidence_in_cycle"] > 0 else state["cycles_without_new_evidence"] + 1
            ),
            "cycles_without_useful_sources": (
                0 if state["useful_source_in_cycle"] else state["cycles_without_useful_sources"] + 1
            ),
            "consecutive_empty_search_cycles": (
                state["consecutive_empty_search_cycles"] + 1
                if state.get("technical_reason") == "no_results"
                else 0
            ),
            "consecutive_technical_failures": (
                state["consecutive_technical_failures"] + 1
                if state.get("technical_reason") in {"no_results", "no_topics", "search_error"}
                else 0
            ),
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    @traceable(name="evaluator-node")
    @log_node_activity("evaluator", "Evaluating: {query}")
    def __call__(self, state: ResearchState) -> dict:
        chapter_id = state.get("current_chapter_id") or ""

        # Deterministic coverage check scoped to current chapter
        chapter_plan = [t for t in state["plan"] if t.chapter_id == chapter_id] if chapter_id else state["plan"]
        # Only evaluate sub-topics when they exist; the depth=0 chapter is a container
        sub_topics = [t for t in chapter_plan if t.depth > 0]
        if sub_topics:
            chapter_plan = sub_topics
        deterministic_resolved, deterministic_gaps = compute_minimum_coverage(
            chapter_plan,
            state["curated_evidence"],
            state["topic_attempts"],
        )

        plan = self._update_topic_statuses(state, deterministic_resolved)
        open_gaps = enrich_gaps_with_search_terms([*deterministic_gaps], plan)
        counter_updates = self._counter_updates(state)

        updated_state = dict(state)
        updated_state["plan"] = plan
        updated_state["open_gaps"] = open_gaps
        updated_state.update(counter_updates)
        synthesis_budget = self._runtime.context_manager.synthesis_budget(
            updated_state  # type: ignore[arg-type]
        )

        stop_reason = self._determine_stop_reason(
            plan=plan,
            synthesis_budget=synthesis_budget,
            current_iteration=state["current_iteration"],
            max_iterations=state["max_iterations"],
            cycles_without_new_evidence=counter_updates["cycles_without_new_evidence"],
            cycles_without_useful_sources=counter_updates["cycles_without_useful_sources"],
            consecutive_technical_failures=counter_updates["consecutive_technical_failures"],
        )

        # Check whether all topics of this chapter are done
        chapter_topics_in_plan = [t for t in plan if t.chapter_id == chapter_id] if chapter_id else plan
        # Only check sub-topics when they exist for completion
        chapter_subs = [t for t in chapter_topics_in_plan if t.depth > 0]
        if chapter_subs:
            chapter_topics_in_plan = chapter_subs
        all_chapter_done = bool(chapter_topics_in_plan) and all(
            t.status in {TopicStatus.COMPLETED, TopicStatus.EXHAUSTED} for t in chapter_topics_in_plan
        )

        log_runtime_event(
            self._runtime,
            "[evaluator] Topic evaluation complete",
            verbosity=1,
            chapter_id=chapter_id,
            stop_reason=stop_reason,
            all_chapter_done=all_chapter_done,
        )
        log_runtime_event(
            self._runtime,
            "[evaluator] Gap snapshot",
            verbosity=2,
            open_gaps=summarize_gaps(open_gaps),
        )

        return {
            "plan": plan,
            "open_gaps": open_gaps,
            **counter_updates,
            "synthesis_budget": synthesis_budget,
            "stop_reason": stop_reason,
            "current_batch": [],
            "extracted_evidence_buffer": [],
            "technical_reason": None if stop_reason else state.get("technical_reason"),
        }
