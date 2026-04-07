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
    ResearchTopic,
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

    def _empty_update(self, state: ResearchState, technical_reason: str) -> dict:
        return {
            "current_batch": [],
            "candidate_queue": [],
            "technical_reason": technical_reason,
            "current_iteration": state["current_iteration"] + 1,
        }

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

    def _mark_active_topic(self, state: ResearchState, topic_id: str) -> list[ResearchTopic]:
        updated: list[ResearchTopic] = []
        for topic in state["plan"]:
            if topic.id == topic_id:
                updated.append(topic.model_copy(update={"status": TopicStatus.IN_PROGRESS}))
            elif topic.status == TopicStatus.IN_PROGRESS:
                updated.append(topic.model_copy(update={"status": TopicStatus.PENDING}))
            else:
                updated.append(topic)
        return updated

    def _select_active_search(
        self,
        state: ResearchState,
    ) -> tuple[ResearchTopic | None, str | None, list[ResearchTopic], dict[str, int]]:
        chapter_id = state.get("current_chapter_id")
        plan_scope = (
            [t for t in state["plan"] if t.chapter_id == chapter_id] if chapter_id else state["plan"]
        )
        # Prefer sub-topics (depth>0) over the chapter container
        sub_topics = [t for t in plan_scope if t.depth > 0]
        if sub_topics:
            plan_scope = sub_topics
        active_topic = choose_active_topic(plan_scope, state["topic_attempts"], state["topic_coverage"])
        if active_topic is None:
            return None, None, state["plan"], dict(state["topic_attempts"])

        query = self._select_query(state, active_topic.id, build_search_query(active_topic))
        plan = self._mark_active_topic(state, active_topic.id)
        topic_attempts = dict(state["topic_attempts"])
        topic_attempts[active_topic.id] = topic_attempts.get(active_topic.id, 0) + 1
        return active_topic, query, plan, topic_attempts

    def _annotate_results(
        self,
        raw_results: list[SearchCandidate],
        topic_id: str,
        query: str,
    ) -> list[SearchCandidate]:
        return [
            result.model_copy(
                update={
                    "topic_ids": list(dict.fromkeys([*result.topic_ids, topic_id])),
                    "query": query,
                }
            )
            for result in raw_results
        ]

    def _validate_candidates(
        self,
        candidates: list[SearchCandidate],
        active_topic: ResearchTopic,
    ) -> tuple[list[SearchCandidate], list[tuple[str, SourceDiscardReason, str]]]:
        filtered: list[tuple[str, SourceDiscardReason, str]] = []
        validated: list[SearchCandidate] = []
        for candidate in candidates:
            is_valid, note = validate_candidate_for_topic(candidate, active_topic)
            if not is_valid:
                filtered.append((candidate.url, SourceDiscardReason.LOW_VALUE, note))
                continue
            validated.append(candidate)
        return validated, filtered

    def _domain_counts(self, state: ResearchState) -> Counter[str]:
        domains = Counter(candidate.domain for candidate in state["candidate_queue"])
        domains.update(extract_domain(record.final_url or record.url) for record in state["visited_urls"].values())
        domains.update(extract_domain(source.url) for source in state["discarded_sources"])
        return domains

    def _rank_candidates(
        self,
        state: ResearchState,
        active_topic: ResearchTopic,
        candidates: list[SearchCandidate],
    ) -> tuple[list[SearchCandidate], list[tuple[str, SourceDiscardReason, str]]]:
        domains = self._domain_counts(state)
        ranked = sorted(
            [score_candidate(candidate, active_topic, state["visited_urls"], domains) for candidate in candidates],
            key=lambda item: item.score,
            reverse=True,
        )
        return prune_queue_by_domain(ranked, state["curated_evidence"])

    def _search_candidates(
        self,
        state: ResearchState,
        active_topic: ResearchTopic,
        query: str,
    ) -> tuple[list[SearchCandidate], list[DiscardedSource], int, int]:
        raw_results = self._runtime.search_client.search(
            query,
            max_results=self._runtime.config.search.results_per_query,
        )
        prepared_results = self._annotate_results(raw_results, active_topic.id, query)
        deduped, discarded, repeated_urls = deduplicate_candidates(
            [*prepared_results, *state["candidate_queue"]],
            state["visited_urls"],
        )
        validated, filtered = self._validate_candidates(deduped, active_topic)
        ranked, pruned = self._rank_candidates(state, active_topic, validated)
        discarded_sources = [
            DiscardedSource(url=url, reason=reason, note=note)
            for url, reason, note in [*discarded, *filtered, *pruned]
        ]
        return ranked, discarded_sources, repeated_urls, len(raw_results)

    def _materialize_batch(
        self,
        state: ResearchState,
        candidates: list[SearchCandidate],
        discarded_sources: list[DiscardedSource],
    ) -> tuple[list[SearchCandidate], list[SearchCandidate], dict[str, SourceRecord], list[DiscardedSource]]:
        visited_urls = dict(state["visited_urls"])
        current_batch: list[SearchCandidate] = []
        remaining_queue: list[SearchCandidate] = []
        all_discarded_sources = [*state["discarded_sources"], *discarded_sources]
        batch_size = self._runtime.config.runtime.search_batch_size

        for candidate in candidates:
            prepared_candidate, discarded_source, source_record = self._prepare_candidate(candidate)
            if discarded_source is not None:
                all_discarded_sources.append(discarded_source)
                visited_urls[source_record.url] = source_record
                continue
            if prepared_candidate is None:
                continue
            if len(current_batch) < batch_size:
                current_batch.append(prepared_candidate)
                visited_urls[source_record.url] = source_record
                continue
            remaining_queue.append(prepared_candidate)

        return current_batch, remaining_queue, visited_urls, all_discarded_sources

    def _build_search_attempt(
        self,
        state: ResearchState,
        active_topic_id: str,
        query: str,
        discovered_urls: int,
        accepted_urls: int,
        repeated_urls: int,
    ) -> SearchAttempt:
        return SearchAttempt(
            topic_id=active_topic_id,
            query=query,
            iteration=state["current_iteration"] + 1,
            discovered_urls=discovered_urls,
            accepted_urls=accepted_urls,
            repeated_urls=repeated_urls,
            empty_result=accepted_urls == 0,
            technical_error=None if accepted_urls else "no_results",
        )

    def _plan_with_last_query(
        self,
        plan: list[ResearchTopic],
        topic_id: str,
        query: str,
    ) -> list[ResearchTopic]:
        return [
            topic.model_copy(update={"last_query": query}) if topic.id == topic_id else topic
            for topic in plan
        ]

    def _no_results_update(
        self,
        state: ResearchState,
        plan: list[ResearchTopic],
        active_topic: ResearchTopic,
        topic_attempts: dict[str, int],
        query: str,
        search_attempt: SearchAttempt,
        discarded_sources: list[DiscardedSource],
        visited_urls: dict[str, SourceRecord],
    ) -> dict:
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
            "new_evidence_in_cycle": 0,
            "merged_evidence_in_cycle": 0,
            "useful_source_in_cycle": False,
        }

    def _success_update(
        self,
        state: ResearchState,
        plan: list[ResearchTopic],
        active_topic: ResearchTopic,
        topic_attempts: dict[str, int],
        query: str,
        search_attempt: SearchAttempt,
        current_batch: list[SearchCandidate],
        remaining_queue: list[SearchCandidate],
        discarded_sources: list[DiscardedSource],
        visited_urls: dict[str, SourceRecord],
        ranked_candidates: list[SearchCandidate],
    ) -> dict:
        log_runtime_event(self._runtime, "[source_manager] Discovered candidate", verbosity=1, url=current_batch[0].url)
        log_runtime_event(
            self._runtime,
            "[source_manager] Ranked search candidates",
            verbosity=3,
            topic_id=active_topic.id,
            query=query,
            top_candidates=summarize_search_candidates(ranked_candidates),
            open_gaps=summarize_gaps(state["open_gaps"]),
        )
        return {
            "plan": self._plan_with_last_query(plan, active_topic.id, query),
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
            "new_evidence_in_cycle": 0,
            "merged_evidence_in_cycle": 0,
            "useful_source_in_cycle": True,
            "extracted_evidence_buffer": [],
        }

    @traceable(name="source-manager-node")
    @log_node_activity("source_manager", "Managing: {query}")
    def __call__(self, state: ResearchState) -> dict:
        active_topic, query, plan, topic_attempts = self._select_active_search(state)
        if active_topic is None or query is None:
            return self._empty_update(state, "no_topics")

        ranked_candidates, discarded_sources, repeated_urls, discovered_urls = self._search_candidates(
            state,
            active_topic,
            query,
        )
        current_batch, remaining_queue, visited_urls, discarded_sources = self._materialize_batch(
            state,
            ranked_candidates,
            discarded_sources,
        )
        search_attempt = self._build_search_attempt(
            state,
            active_topic.id,
            query,
            discovered_urls,
            len(current_batch),
            repeated_urls,
        )

        if not current_batch:
            return self._no_results_update(
                state,
                plan,
                active_topic,
                topic_attempts,
                query,
                search_attempt,
                discarded_sources,
                visited_urls,
            )

        return self._success_update(
            state,
            plan,
            active_topic,
            topic_attempts,
            query,
            search_attempt,
            current_batch,
            remaining_queue,
            discarded_sources,
            visited_urls,
            ranked_candidates,
        )
