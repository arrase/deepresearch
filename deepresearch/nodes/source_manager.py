"""Topic selection and single-source search node."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import (
    build_search_query,
    choose_active_topic,
    classify_source_content,
    deduplicate_candidates,
    extract_domain,
    prune_queue_by_domain,
    reformulate_queries,
    sanitize_source_title,
    score_candidate,
    split_source_content,
    summarize_gaps,
    summarize_search_candidates,
    validate_candidate_for_topic,
)
from ..state import (
    DiscardedSource,
    ResearchState,
    SearchAttempt,
    SearchCandidate,
    SourceDiscardReason,
    SourceRecord,
    TopicStatus,
)
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class SourceManagerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _prepare_candidate(
        self,
        candidate: SearchCandidate,
    ) -> tuple[SearchCandidate | None, DiscardedSource | None, SourceRecord]:
        raw_content, diagnostics = split_source_content(
            candidate.raw_content or "",
            max_chars=self._runtime.config.search.max_raw_content_chars,
        )
        url = candidate.normalized_url or candidate.url
        title = sanitize_source_title(candidate.title, url) or "Unknown Source"
        reason = classify_source_content(
            content=raw_content,
            diagnostics=diagnostics,
            min_source_chars=self._runtime.config.search.min_source_chars,
        )
        record = SourceRecord(
            url=url,
            final_url=url,
            title=title,
            extracted_chars=len(raw_content),
            topic_ids=list(candidate.topic_ids),
            last_error=diagnostics or (reason.value if reason is not None else None),
        )
        if reason is not None:
            discarded = DiscardedSource(
                url=url,
                reason=reason,
                note=diagnostics or reason.value,
            )
            return None, discarded, record
        prepared = candidate.model_copy(update={"raw_content": raw_content, "title": title})
        return prepared, None, record

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
        filtered: list[tuple[str, SourceDiscardReason, str]] = []
        validated = []
        for candidate in deduped:
            is_valid, note = validate_candidate_for_topic(candidate, active_topic)
            if not is_valid:
                filtered.append((candidate.url, SourceDiscardReason.LOW_VALUE, note))
                continue
            validated.append(candidate)
        domains = Counter(candidate.domain for candidate in state["candidate_queue"])
        domains.update(extract_domain(record.final_url or record.url) for record in state["visited_urls"].values())
        domains.update(extract_domain(source.url) for source in state["discarded_sources"])
        ranked = sorted(
            [score_candidate(candidate, active_topic, state["visited_urls"], domains) for candidate in validated],
            key=lambda item: item.score,
            reverse=True,
        )
        ranked, pruned = prune_queue_by_domain(ranked, state["curated_evidence"])
        visited_urls = dict(state["visited_urls"])
        batch_size = self._runtime.config.runtime.search_batch_size
        current_batch: list[SearchCandidate] = []
        remaining_queue: list[SearchCandidate] = []
        discarded_sources = [
            *state["discarded_sources"],
            *[
                DiscardedSource(url=url, reason=reason, note=note)
                for url, reason, note in [*discarded, *filtered, *pruned]
            ],
        ]
        for candidate in ranked:
            prepared_candidate, discarded_source, source_record = self._prepare_candidate(candidate)
            if discarded_source is not None:
                discarded_sources.append(discarded_source)
                visited_urls[source_record.url] = source_record
                continue
            if prepared_candidate is None:
                continue
            if len(current_batch) < batch_size:
                current_batch.append(prepared_candidate)
                visited_urls[source_record.url] = source_record
            else:
                remaining_queue.append(prepared_candidate)
        search_attempt = SearchAttempt(
            topic_id=active_topic.id,
            query=query,
            iteration=state["current_iteration"] + 1,
            discovered_urls=len(raw_results),
            accepted_urls=len(current_batch),
            repeated_urls=repeated_urls,
            empty_result=not current_batch,
            technical_error=None if current_batch else "no_results",
        )

        if not current_batch:
            log_runtime_event(
                self._runtime,
                "[source_manager] Search produced no usable candidates",
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
                "visited_urls": visited_urls,
                "candidate_queue": [],
                "current_batch": [],
                "current_iteration": state["current_iteration"] + 1,
                "technical_reason": "no_results",
                "replan_requested": self._runtime.config.runtime.allow_dynamic_replan,
                "new_evidence_in_cycle": 0,
                "merged_evidence_in_cycle": 0,
                "useful_source_in_cycle": False,
            }

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
            "visited_urls": visited_urls,
            "candidate_queue": remaining_queue,
            "current_batch": current_batch,
            "discarded_sources": discarded_sources,
            "current_iteration": state["current_iteration"] + 1,
            "technical_reason": None,
            "replan_requested": False,
            "new_evidence_in_cycle": 0,
            "merged_evidence_in_cycle": 0,
            "useful_source_in_cycle": True,
            "extracted_evidence_buffer": [],
        }
