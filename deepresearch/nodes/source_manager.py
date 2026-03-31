"""Source manager node implementation."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ..core.utils import deduplicate_candidates, extract_domain, score_candidate
from ..state import DiscardedSource, ResearchState, SearchCandidate, SourceDiscardReason, Subquery
from .base import record_telemetry


class SourceManagerNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("source_manager", "Managing research sources for: {query}")
    def __call__(self, state: ResearchState) -> dict:
        if state["search_queue"]:
            next_candidate = state["search_queue"][0]
            event = self._runtime.telemetry.record(
                "source_manager",
                "Reusing an already prioritized candidate",
                url=next_candidate.url,
            )
            return {
                "search_queue": state["search_queue"][1:],
                "current_candidate": next_candidate,
                "iteration": state["iteration"] + 1,
                "telemetry": [*state["telemetry"], event],
            }

        candidate_queries: list[str] = []
        # Prioritize open gaps (newest first)
        for gap in reversed(state["open_gaps"]):
            for query in gap.suggested_queries:
                if query not in state["completed_search_queries"] and query not in candidate_queries:
                    candidate_queries.append(query)
        
        # Then newer search intents from iterative planning (deep-dives)
        for intent in reversed(state["search_intents"]):
            if intent.query not in state["completed_search_queries"] and intent.query not in candidate_queries:
                candidate_queries.append(intent.query)
        
        # Fallback to active subqueries
        if not candidate_queries:
            for subquery in reversed(state["active_subqueries"]):
                if subquery.question not in state["completed_search_queries"] and subquery.question not in candidate_queries:
                    candidate_queries.append(subquery.question)

        candidate_queries = candidate_queries[: self._runtime.config.search.max_queries_per_cycle]
        raw_candidates: list[SearchCandidate] = []
        completed = [*state["completed_search_queries"]]
        for query in candidate_queries:
            self._runtime.telemetry.record(
                "source_manager",
                "Querying search backend",
                query=query,
                backend=self._runtime.config.search.backend,
            )
            try:
                results = self._runtime.search_client.search(
                    query,
                    max_results=self._runtime.config.search.results_per_query,
                )
            except Exception as exc:  # noqa: BLE001
                event = self._runtime.telemetry.record(
                    "source_manager",
                    "Search backend failed",
                    query=query,
                    error=str(exc),
                )
                return {
                    "completed_search_queries": [*completed, query],
                    "telemetry": [*state["telemetry"], event],
                    "fallback_reason": state["fallback_reason"] or "search_backend_failure",
                }
            completed.append(query)
            for candidate in results:
                if not candidate.subquery_ids:
                    candidate.subquery_ids = self._match_subqueries(candidate, state["active_subqueries"])
                candidate.domain = candidate.domain or extract_domain(candidate.url)
                raw_candidates.append(candidate)

        deduped, discarded = deduplicate_candidates(raw_candidates, visited_urls=state["visited_urls"])
        discarded_sources = [*state["discarded_sources"]]
        for url, reason, note in discarded:
            discarded_sources.append(DiscardedSource(url=url, reason=reason, note=note))

        domain_counts = Counter(entry.domain for entry in state["search_queue"])
        ranked = [
            score_candidate(
                candidate,
                active_subqueries=state["active_subqueries"],
                visited_urls=state["visited_urls"],
                domain_counts=domain_counts,
            )
            for candidate in deduped
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        if not ranked:
            bootstrap = self._bootstrap_candidates(state)
            if bootstrap:
                bootstrap, bootstrap_discarded = deduplicate_candidates(bootstrap, visited_urls=state["visited_urls"])
                for url, reason, note in bootstrap_discarded:
                    discarded_sources.append(DiscardedSource(url=url, reason=reason, note=note))
                bootstrap_ranked = [
                    score_candidate(
                        candidate,
                        active_subqueries=state["active_subqueries"],
                        visited_urls=state["visited_urls"],
                        domain_counts=domain_counts,
                    )
                    for candidate in bootstrap
                ]
                bootstrap_ranked.sort(key=lambda item: item.score, reverse=True)
                if bootstrap_ranked:
                    next_candidate = bootstrap_ranked[0]
                    event = self._runtime.telemetry.record(
                        "source_manager",
                        "Activating authoritative bootstrap sources",
                        selected=next_candidate.url,
                        candidates=len(bootstrap_ranked),
                    )
                    return {
                        "completed_search_queries": completed,
                        "discarded_sources": discarded_sources,
                        "search_queue": bootstrap_ranked[1:],
                        "current_candidate": next_candidate,
                        "iteration": state["iteration"] + 1,
                        "telemetry": [*state["telemetry"], event],
                    }
            fallback_reason = state["fallback_reason"] or "no_actionable_sources"
            event = self._runtime.telemetry.record(
                "source_manager",
                "No new actionable sources are available",
            )
            return {
                "completed_search_queries": completed,
                "discarded_sources": discarded_sources,
                "current_candidate": None,
                "iteration": state["iteration"] + 1,
                "fallback_reason": fallback_reason,
                "telemetry": [*state["telemetry"], event],
            }

        next_candidate = ranked[0]
        event = self._runtime.telemetry.record(
            "source_manager",
            "Sources discovered and prioritized",
            selected=next_candidate.url,
            candidates=len(ranked),
        )
        return {
            "completed_search_queries": completed,
            "discarded_sources": discarded_sources,
            "search_queue": ranked[1:],
            "current_candidate": next_candidate,
            "iteration": state["iteration"] + 1,
            "telemetry": [*state["telemetry"], event],
        }

    def _match_subqueries(self, candidate: SearchCandidate, subqueries: list[Subquery]) -> list[str]:
        matched = []
        haystack = f"{candidate.title} {candidate.snippet}".lower()
        for sq in subqueries:
            terms = [token.lower() for token in sq.search_terms or sq.question.split()]
            if any(term in haystack for term in terms[:6]):
                matched.append(sq.id)
        return matched

    def _bootstrap_candidates(self, state: ResearchState) -> list[SearchCandidate]:
        if state["iteration"] > 0:
            return []
        
        # This is a fallback if the first search fails or yields no results
        # We can try to bootstrap from the initial subqueries
        return [
            SearchCandidate(
                url=f"https://www.google.com/search?q={sq.question}",
                title=sq.question,
                snippet="Direct search fallback",
                discovered_via="bootstrap",
                subquery_ids=[sq.id],
            )
            for sq in state["active_subqueries"][:2]
        ]
