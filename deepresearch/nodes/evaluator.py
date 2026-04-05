"""Evaluator node implementation.

The evaluator now separates deterministic evaluation (always runs) from
semantic LLM-based evaluation (only runs periodically or pre-synthesis).
Topics require ``min_attempts_before_exhaustion`` failed cycles before
being marked EXHAUSTED, and the node can trigger replan instead of stopping
when all topics are exhausted but iteration budget remains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.payloads import CoveragePayload
from ..core.utils import compute_minimum_coverage, enrich_gaps_with_search_terms, summarize_gaps
from ..state import ResearchState, ResearchTopic, StopReason, SynthesisBudget, TopicStatus
from .base import log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class EvaluatorNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _should_run_semantic_eval(self, state: ResearchState) -> bool:
        """Decide whether to invoke the LLM evaluator this cycle."""
        interval = self._runtime.config.runtime.semantic_eval_interval
        # interval == 0 means "never automatically, only pre-synthesis"
        if interval <= 0:
            return False
        return state["current_iteration"] % interval == 0

    def _determine_stop_reason_from_values(
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
        if plan and all(topic.status in {TopicStatus.COMPLETED, TopicStatus.EXHAUSTED} for topic in plan):
            return StopReason.PLAN_COMPLETED.value
        if current_iteration >= max_iterations:
            return StopReason.MAX_ITERATIONS_REACHED.value
        runtime = self._runtime.config.runtime
        stuck_by_evidence = cycles_without_new_evidence >= runtime.max_cycles_without_new_evidence
        stuck_by_sources = cycles_without_useful_sources >= runtime.max_cycles_without_useful_sources
        stuck_by_technical = consecutive_technical_failures >= runtime.max_consecutive_technical_failures
        if stuck_by_technical or (stuck_by_evidence and stuck_by_sources):
            return StopReason.STUCK_NO_SOURCES.value
        return None

    def _run_semantic_evaluation(self, state: ResearchState) -> tuple[CoveragePayload | None, dict[str, int], bool]:
        if not self._should_run_semantic_eval(state):
            return None, {}, False
        semantic, usage = self._runtime.llm_workers.evaluate_coverage_with_usage(
            self._runtime.context_manager.evaluator_context(state)
        )
        return semantic, usage, True

    def _force_semantic_evaluation(self, state: ResearchState) -> tuple[CoveragePayload, dict[str, int]]:
        return self._runtime.llm_workers.evaluate_coverage_with_usage(
            self._runtime.context_manager.evaluator_context(state)
        )

    def _update_topic_statuses(
        self,
        state: ResearchState,
        deterministic_resolved: list[str],
        semantic: CoveragePayload | None,
    ) -> tuple[list[ResearchTopic], bool]:
        coverage = state["topic_coverage"]
        plan: list[ResearchTopic] = []
        replan_requested = False
        min_attempts = self._runtime.config.runtime.min_attempts_before_exhaustion
        active_topic_id = state.get("active_topic_id")
        semantic_sufficient = semantic.is_sufficient if semantic else True

        for topic in state["plan"]:
            topic_coverage = coverage.get(topic.id)
            accepted_count = topic_coverage.accepted_evidence_count if topic_coverage else 0
            attempts = state["topic_attempts"].get(topic.id, 0)
            completed = topic.id in deterministic_resolved and (semantic_sufficient or not topic.success_criteria)
            exhausted = (
                topic.id == active_topic_id
                and accepted_count == 0
                and attempts >= min_attempts
                and state.get("technical_reason") in {"no_results", "no_topics", "search_error"}
                and self._runtime.config.runtime.allow_dynamic_replan
            )
            if completed:
                plan.append(topic.model_copy(update={"status": TopicStatus.COMPLETED}))
            elif exhausted:
                plan.append(topic.model_copy(update={"status": TopicStatus.EXHAUSTED}))
                replan_requested = True
            elif topic.id == active_topic_id:
                plan.append(topic.model_copy(update={"status": TopicStatus.IN_PROGRESS}))
            else:
                plan.append(topic)
        return plan, replan_requested

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

    @traceable(name="evaluator-node")
    @log_node_activity("evaluator", "Evaluating: {query}")
    def __call__(self, state: ResearchState) -> dict:
        deterministic_resolved, deterministic_gaps = compute_minimum_coverage(
            state["plan"],
            state["curated_evidence"],
            state["topic_attempts"],
        )

        semantic, usage, semantic_ran = self._run_semantic_evaluation(state)
        active_topic_id = state.get("active_topic_id")
        plan, replan_requested = self._update_topic_statuses(state, deterministic_resolved, semantic)

        semantic_gaps = semantic.open_gaps if semantic else []
        open_gaps = enrich_gaps_with_search_terms([*deterministic_gaps, *semantic_gaps], plan)
        counter_updates = self._counter_updates(state)
        synthesis_budget = self._runtime.context_manager.synthesis_budget(
            {
                **state,
                "plan": plan,
                "open_gaps": open_gaps,
                "cycles_without_new_evidence": counter_updates["cycles_without_new_evidence"],
                "cycles_without_useful_sources": counter_updates["cycles_without_useful_sources"],
                "consecutive_empty_search_cycles": counter_updates["consecutive_empty_search_cycles"],
                "consecutive_technical_failures": counter_updates["consecutive_technical_failures"],
            }
        )

        stop_reason = self._determine_stop_reason_from_values(
            plan=plan,
            synthesis_budget=synthesis_budget,
            current_iteration=state["current_iteration"],
            max_iterations=state["max_iterations"],
            cycles_without_new_evidence=counter_updates["cycles_without_new_evidence"],
            cycles_without_useful_sources=counter_updates["cycles_without_useful_sources"],
            consecutive_technical_failures=counter_updates["consecutive_technical_failures"],
        )

        if stop_reason and not semantic_ran:
            semantic, usage = self._force_semantic_evaluation(state)
            semantic_ran = True
            semantic_gaps = semantic.open_gaps
            open_gaps = enrich_gaps_with_search_terms([*deterministic_gaps, *semantic_gaps], plan)

        if (
            stop_reason == StopReason.PLAN_COMPLETED.value
            and state["current_iteration"] < state["max_iterations"]
            and self._runtime.config.runtime.allow_dynamic_replan
            and not state["curated_evidence"]
        ):
            stop_reason = None
            replan_requested = True

        contradictions = semantic.contradictions if semantic else []
        llm_usage = update_stage_llm_usage(state.get("llm_usage", {}), "evaluator", usage, include_empty=False)

        log_runtime_event(
            self._runtime,
            "[evaluator] Topic evaluation complete",
            verbosity=1,
            active_topic_id=active_topic_id,
            stop_reason=stop_reason,
            replan_requested=replan_requested,
            semantic_eval_ran=semantic_ran,
            **usage,
        )
        log_runtime_event(
            self._runtime,
            "[evaluator] Gap snapshot",
            verbosity=2,
            open_gaps=summarize_gaps(open_gaps),
            semantic_rationale=semantic.rationale if semantic else "skipped",
        )
        return {
            "plan": plan,
            "open_gaps": open_gaps,
            "contradictions": contradictions,
            **counter_updates,
            "synthesis_budget": synthesis_budget,
            "stop_reason": stop_reason,
            "replan_requested": False if stop_reason else replan_requested,
            "llm_usage": llm_usage,
            "current_batch": [],
            "extracted_evidence_buffer": [],
            "technical_reason": None if stop_reason else state.get("technical_reason"),
        }
