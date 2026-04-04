"""Evaluator node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import compute_minimum_coverage, enrich_gaps_with_search_terms, summarize_gaps
from ..state import ResearchState, StopReason, TopicStatus
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class EvaluatorNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _determine_stop_reason(self, state: ResearchState) -> str | None:
        budget = state["synthesis_budget"]
        if budget.final_context_full:
            return StopReason.CONTEXT_SATURATION.value
        if state["plan"] and all(
            topic.status in {TopicStatus.COMPLETED, TopicStatus.EXHAUSTED}
            for topic in state["plan"]
        ):
            return StopReason.PLAN_COMPLETED.value
        if state["current_iteration"] >= state["max_iterations"]:
            return StopReason.MAX_ITERATIONS_REACHED.value
        runtime = self._runtime.config.runtime
        stuck_by_evidence = state["cycles_without_new_evidence"] >= runtime.max_cycles_without_new_evidence
        stuck_by_sources = state["cycles_without_useful_sources"] >= runtime.max_cycles_without_useful_sources
        stuck_by_technical = state["consecutive_technical_failures"] >= runtime.max_consecutive_technical_failures
        if stuck_by_technical or (stuck_by_evidence and stuck_by_sources):
            return StopReason.STUCK_NO_SOURCES.value
        return None

    @traceable(name="evaluator-node")
    @log_node_activity("evaluator", "Evaluating: {query}")
    def __call__(self, state: ResearchState) -> dict:
        deterministic_resolved, deterministic_gaps = compute_minimum_coverage(
            state["plan"],
            state["curated_evidence"],
            state["topic_attempts"],
        )
        semantic, usage = self._runtime.llm_workers.evaluate_coverage_with_usage(
            self._runtime.context_manager.evaluator_context(state)
        )
        coverage = state["topic_coverage"]
        plan = []
        replan_requested = False
        for topic in state["plan"]:
            topic_coverage = coverage.get(topic.id)
            accepted_count = topic_coverage.accepted_evidence_count if topic_coverage else 0
            completed = topic.id in deterministic_resolved and (semantic.is_sufficient or not topic.success_criteria)
            exhausted = (
                topic.id == state.get("active_topic_id")
                and accepted_count == 0
                and state.get("technical_reason") in {"no_results", "no_topics", "browser_error"}
                and self._runtime.config.runtime.allow_dynamic_replan
            )
            if completed:
                plan.append(topic.model_copy(update={"status": TopicStatus.COMPLETED}))
            elif exhausted:
                plan.append(topic.model_copy(update={"status": TopicStatus.EXHAUSTED}))
                replan_requested = True
            elif topic.id == state.get("active_topic_id"):
                plan.append(topic.model_copy(update={"status": TopicStatus.IN_PROGRESS}))
            else:
                plan.append(topic)

        open_gaps = enrich_gaps_with_search_terms([*deterministic_gaps, *semantic.open_gaps], plan)
        cycles_without_new_evidence = (
            0 if state["new_evidence_in_cycle"] > 0 else state["cycles_without_new_evidence"] + 1
        )
        cycles_without_useful_sources = (
            0 if state["useful_source_in_cycle"] else state["cycles_without_useful_sources"] + 1
        )
        consecutive_empty_search_cycles = (
            state["consecutive_empty_search_cycles"] + 1 if state.get("technical_reason") == "no_results" else 0
        )
        consecutive_technical_failures = (
            state["consecutive_technical_failures"] + 1
            if state.get("technical_reason") in {"no_results", "no_topics", "search_error", "browser_error"}
            else 0
        )
        synthesis_budget = self._runtime.context_manager.synthesis_budget(
            {**state, "plan": plan, "open_gaps": open_gaps}
        )
        stop_reason = self._determine_stop_reason(
            {
                **state,
                "plan": plan,
                "open_gaps": open_gaps,
                "cycles_without_new_evidence": cycles_without_new_evidence,
                "cycles_without_useful_sources": cycles_without_useful_sources,
                "consecutive_empty_search_cycles": consecutive_empty_search_cycles,
                "consecutive_technical_failures": consecutive_technical_failures,
                "synthesis_budget": synthesis_budget,
            }
        )
        llm_usage = {**state.get("llm_usage", {}), "evaluator": usage}
        log_runtime_event(
            self._runtime,
            "[evaluator] Topic evaluation complete",
            verbosity=1,
            active_topic_id=state.get("active_topic_id"),
            stop_reason=stop_reason,
            replan_requested=replan_requested,
            **usage,
        )
        log_runtime_event(
            self._runtime,
            "[evaluator] Gap snapshot",
            verbosity=2,
            open_gaps=summarize_gaps(open_gaps),
            semantic_rationale=semantic.rationale,
        )
        return {
            "plan": plan,
            "open_gaps": open_gaps,
            "contradictions": semantic.contradictions,
            "cycles_without_new_evidence": cycles_without_new_evidence,
            "cycles_without_useful_sources": cycles_without_useful_sources,
            "consecutive_empty_search_cycles": consecutive_empty_search_cycles,
            "consecutive_technical_failures": consecutive_technical_failures,
            "synthesis_budget": synthesis_budget,
            "stop_reason": stop_reason,
            "replan_requested": False if stop_reason else replan_requested,
            "llm_usage": llm_usage,
            "current_batch": [],
            "current_browser_result": None,
            "extracted_evidence_buffer": [],
            "technical_reason": None if stop_reason else state.get("technical_reason"),
        }
