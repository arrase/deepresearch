"""Source manager node implementation."""

from __future__ import annotations
from collections import Counter
from typing import Any
from ..core.utils import deduplicate_candidates, score_candidate
from ..state import DiscardedSource, ResearchState
from .base import record_telemetry

class SourceManagerNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("source_manager", "Managing: {query}")
    def __call__(self, state: ResearchState) -> dict:
        if state["search_queue"]:
            next_c = state["search_queue"][0]
            event = self._runtime.telemetry.record("source_manager", "Queue next", url=next_c.url)
            return {
                "search_queue": state["search_queue"][1:],
                "current_candidate": next_c,
                "current_browser_result": None,
                "latest_evidence": [],
                "progress_score": 0,
                "iteration": state["iteration"] + 1,
                "technical_reason": None,
                "telemetry": [*state["telemetry"], event],
            }

        # Collect queries: Gaps > Intents > Subqueries
        queries = [q for g in reversed(state["open_gaps"]) for q in g.suggested_queries]
        queries += [i.query for i in reversed(state["search_intents"])]
        queries += [s.question for s in reversed(state["active_subqueries"])]
        queries = [q for q in queries if q not in state["completed_search_queries"]][:self._runtime.config.search.max_queries_per_cycle]

        if not queries:
            event = self._runtime.telemetry.record("source_manager", "No queries left")
            return {
                "iteration": state["iteration"] + 1,
                "current_candidate": None,
                "current_browser_result": None,
                "latest_evidence": [],
                "progress_score": 0,
                "technical_reason": "no_queries",
                "telemetry": [*state["telemetry"], event],
            }

        raw = []
        for q in queries:
            try:
                raw.extend(self._runtime.search_client.search(q))
            except Exception as e:
                event = self._runtime.telemetry.record("source_manager", "Search error", q=q, err=str(e))
                return {
                    "completed_search_queries": [*state["completed_search_queries"], q],
                    "current_candidate": None,
                    "current_browser_result": None,
                    "latest_evidence": [],
                    "progress_score": 0,
                    "telemetry": [*state["telemetry"], event],
                    "technical_reason": "search_error",
                }

        deduped, discarded = deduplicate_candidates(raw, state["visited_urls"])
        new_discarded = state["discarded_sources"] + [DiscardedSource(url=u, reason=r, note=n) for u, r, n in discarded]

        domains = Counter(c.domain for c in state["search_queue"])
        ranked = sorted([score_candidate(c, state["active_subqueries"], state["visited_urls"], domains) for c in deduped], key=lambda x: x.score, reverse=True)

        if not ranked:
            event = self._runtime.telemetry.record("source_manager", "No results")
            return {
                "completed_search_queries": [*state["completed_search_queries"], *queries],
                "discarded_sources": new_discarded,
                "current_candidate": None,
                "current_browser_result": None,
                "latest_evidence": [],
                "progress_score": 0,
                "iteration": state["iteration"] + 1,
                "technical_reason": "no_results",
                "telemetry": [*state["telemetry"], event],
            }

        next_c = ranked[0]
        event = self._runtime.telemetry.record("source_manager", "Discovered", url=next_c.url, count=len(ranked))
        return {
            "completed_search_queries": [*state["completed_search_queries"], *queries],
            "discarded_sources": new_discarded,
            "search_queue": ranked[1:],
            "current_candidate": next_c,
            "current_browser_result": None,
            "latest_evidence": [],
            "progress_score": 0,
            "iteration": state["iteration"] + 1,
            "technical_reason": None,
            "telemetry": [*state["telemetry"], event],
        }
