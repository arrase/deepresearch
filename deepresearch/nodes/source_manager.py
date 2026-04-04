"""Topic selection and single-source search node."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import (
    build_search_query,
    choose_active_topic,
    deduplicate_candidates,
    prune_queue_by_domain,
    reformulate_queries,
    score_candidate,
    summarize_gaps,
    summarize_search_candidates,
)
from ..state import DiscardedSource, ResearchState, SearchAttempt, TopicStatus
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class SourceManagerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _select_query(self, state: ResearchState, topic_id: str, fallback_query: str) -> str:
        for intent in reversed(state["search_intents"]):
            if topic_id in intent.topic_ids and intent.query not in state["completed_search_queries"]:
                return intent.query
        if fallback_query not in state["completed_search_queries"]:
            return fallback_query
        variants = reformulate_queries(fallback_query, set(state["completed_search_queries"]), fallback_query.split())
        for variant in variants:
            if variant not in state["completed_search_queries"]:
                return variant
        return fallback_query

    def _mark_active_topic(self, state: ResearchState, topic_id: str) -> list:
        updated = []
        for topic in state["plan"]:
            if topic.id == topic_id:
                updated.append(topic.model_copy(update={"status": TopicStatus.IN_PROGRESS}))
            elif topic.status == TopicStatus.IN_PROGRESS:
                updated.append(topic.model_copy(update={"status": TopicStatus.PENDING}))
            else:
                updated.append(topic)
        return updated

    @traceable(name="source-manager-node")
    @log_node_activity("source_manager", "Managing: {query}")
    def __call__(self, state: ResearchState) -> dict:
        active_topic = choose_active_topic(state["plan"], state["topic_attempts"], state["topic_coverage"])
        if active_topic is None:
            return {
                "current_batch": [],
                "candidate_queue": [],
                "technical_reason": "no_topics",
                "replan_requested": False,
                "current_iteration": state["current_iteration"] + 1,
            }

        query = self._select_query(state, active_topic.id, build_search_query(active_topic))
        plan = self._mark_active_topic(state, active_topic.id)
        topic_attempts = dict(state["topic_attempts"])
        topic_attempts[active_topic.id] = topic_attempts.get(active_topic.id, 0) + 1

        raw_results = self._runtime.search_client.search(
            query,
            max_results=self._runtime.config.search.results_per_query,
        )
        prepared_results = [
            result.model_copy(
                update={
                    "topic_ids": list(dict.fromkeys([*result.topic_ids, active_topic.id])),
                    "query": query,
                }
            )
            for result in raw_results
        ]
        deduped, discarded, repeated_urls = deduplicate_candidates(
            [*prepared_results, *state["candidate_queue"]],
            state["visited_urls"],
        )
        domains = Counter(candidate.domain for candidate in state["candidate_queue"])
        ranked = sorted(
            [score_candidate(candidate, active_topic, state["visited_urls"], domains) for candidate in deduped],
            key=lambda item: item.score,
            reverse=True,
        )
        ranked, pruned = prune_queue_by_domain(ranked, state["curated_evidence"])
        discarded_sources = [
            *state["discarded_sources"],
            *[
                DiscardedSource(url=url, reason=reason, note=note)
                for url, reason, note in [*discarded, *pruned]
            ],
        ]
        search_attempt = SearchAttempt(
            topic_id=active_topic.id,
            query=query,
            iteration=state["current_iteration"] + 1,
            discovered_urls=len(raw_results),
            accepted_urls=min(len(ranked), self._runtime.config.runtime.search_batch_size),
            repeated_urls=repeated_urls,
            empty_result=not ranked,
            technical_error=None if ranked else "no_results",
        )

        if not ranked:
            log_runtime_event(
                self._runtime,
                "[source_manager] Search produced no new candidates",
                verbosity=1,
                topic_id=active_topic.id,
                query=query,
            )
            return {
                "plan": plan,
                "active_topic_id": active_topic.id,
                "topic_attempts": topic_attempts,
                "search_history": [*state["search_history"], search_attempt],
                "completed_search_queries": [*state["completed_search_queries"], query],
                "failed_queries": [*state["failed_queries"], query],
                "discarded_sources": discarded_sources,
                "candidate_queue": [],
                "current_batch": [],
                "current_iteration": state["current_iteration"] + 1,
                "technical_reason": "no_results",
                "replan_requested": self._runtime.config.runtime.allow_dynamic_replan,
                "new_evidence_in_cycle": 0,
                "merged_evidence_in_cycle": 0,
                "useful_source_in_cycle": False,
            }

        batch_size = self._runtime.config.runtime.search_batch_size
        current_batch = ranked[:batch_size]
        remaining_queue = ranked[batch_size:]
        log_runtime_event(self._runtime, "[source_manager] Discovered candidate", verbosity=1, url=current_batch[0].url)
        log_runtime_event(
            self._runtime,
            "[source_manager] Ranked search candidates",
            verbosity=3,
            topic_id=active_topic.id,
            query=query,
            top_candidates=summarize_search_candidates(ranked),
            open_gaps=summarize_gaps(state["open_gaps"]),
        )
        return {
            "plan": [
                topic.model_copy(update={"last_query": query}) if topic.id == active_topic.id else topic
                for topic in plan
            ],
            "active_topic_id": active_topic.id,
            "topic_attempts": topic_attempts,
            "search_history": [*state["search_history"], search_attempt],
            "completed_search_queries": [*state["completed_search_queries"], query],
            "candidate_queue": remaining_queue,
            "current_batch": current_batch,
            "discarded_sources": discarded_sources,
            "current_iteration": state["current_iteration"] + 1,
            "technical_reason": None,
            "replan_requested": False,
            "new_evidence_in_cycle": 0,
            "merged_evidence_in_cycle": 0,
            "useful_source_in_cycle": False,
            "current_browser_result": None,
            "extracted_evidence_buffer": [],
        }
