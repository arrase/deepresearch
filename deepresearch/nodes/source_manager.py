"""Source manager node implementation."""

from __future__ import annotations
from collections import Counter
from typing import Any
from ..core.utils import deduplicate_candidates, score_candidate, summarize_gaps, summarize_search_candidates
from ..state import DiscardedSource, ResearchState
from .base import record_telemetry

class SourceManagerNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    def _build_query_specs(self, state: ResearchState) -> list[dict[str, object]]:
        query_specs: dict[str, dict[str, object]] = {}

        def register_query(query: str, subquery_ids: list[str], discovered_via: str) -> None:
            if not query.strip() or query in state["completed_search_queries"]:
                return
            spec = query_specs.get(query)
            if spec is None:
                query_specs[query] = {
                    "query": query,
                    "subquery_ids": list(dict.fromkeys(subquery_ids)),
                    "discovered_via": discovered_via,
                }
                return
            spec["subquery_ids"] = list(dict.fromkeys([*spec["subquery_ids"], *subquery_ids]))

        for gap in reversed(state["open_gaps"]):
            for query in gap.suggested_queries:
                register_query(query, [gap.subquery_id], "gap")
        for intent in reversed(state["search_intents"]):
            register_query(intent.query, intent.subquery_ids, "intent")
        for subquery in reversed(state["active_subqueries"]):
            register_query(subquery.question, [subquery.id], "subquery")

        return list(query_specs.values())

    @record_telemetry("source_manager", "Managing: {query}")
    def __call__(self, state: ResearchState) -> dict:
        query_specs = self._build_query_specs(state)
        gap_query_specs = [spec for spec in query_specs if spec["discovered_via"] == "gap"]

        if state["search_queue"] and not gap_query_specs:
            next_c = state["search_queue"][0]
            event = self._runtime.telemetry.record("source_manager", "Queue next", verbosity=1, payload_type="queue", url=next_c.url)
            detail_event = self._runtime.telemetry.record(
                "source_manager",
                "Selected queued candidate",
                verbosity=3,
                payload_type="web_candidates",
                candidate=summarize_search_candidates([next_c], limit=1),
                remaining_queue=len(state["search_queue"]) - 1,
            )
            return {
                "search_queue": state["search_queue"][1:],
                "current_candidate": next_c,
                "current_browser_result": None,
                "latest_evidence": [],
                "progress_score": 0,
                "iteration": state["iteration"] + 1,
                "technical_reason": None,
                "telemetry": self._runtime.telemetry.extend(state["telemetry"], event, detail_event),
            }

        selected_specs = (gap_query_specs or query_specs)[:self._runtime.config.search.max_queries_per_cycle]
        queries = [str(spec["query"]) for spec in selected_specs]

        if not selected_specs:
            event = self._runtime.telemetry.record("source_manager", "No queries left", verbosity=1, payload_type="queue")
            detail_event = self._runtime.telemetry.record(
                "source_manager",
                "No actionable queries available",
                verbosity=3,
                payload_type="web_candidates",
                open_gaps=summarize_gaps(state["open_gaps"]),
                remaining_intents=[intent.query for intent in state["search_intents"][:5]],
            )
            return {
                "iteration": state["iteration"] + 1,
                "current_candidate": None,
                "current_browser_result": None,
                "latest_evidence": [],
                "progress_score": 0,
                "technical_reason": "no_queries",
                "telemetry": self._runtime.telemetry.extend(state["telemetry"], event, detail_event),
            }

        raw = []
        for spec in selected_specs:
            query = str(spec["query"])
            subquery_ids = list(spec["subquery_ids"])
            try:
                results = self._runtime.search_client.search(query)
                for candidate in results:
                    candidate.subquery_ids = list(dict.fromkeys([*candidate.subquery_ids, *subquery_ids]))
                    if candidate.discovered_via == "search":
                        candidate.discovered_via = str(spec["discovered_via"])
                raw.extend(results)
            except Exception as e:
                event = self._runtime.telemetry.record("source_manager", "Search error", verbosity=1, payload_type="error", q=query, err=str(e))
                return {
                    "completed_search_queries": [*state["completed_search_queries"], query],
                    "current_candidate": None,
                    "current_browser_result": None,
                    "latest_evidence": [],
                    "progress_score": 0,
                    "telemetry": self._runtime.telemetry.extend(state["telemetry"], event),
                    "technical_reason": "search_error",
                }

        candidate_pool = [*raw, *state["search_queue"]]
        deduped, discarded = deduplicate_candidates(candidate_pool, state["visited_urls"])
        new_discarded = state["discarded_sources"] + [DiscardedSource(url=u, reason=r, note=n) for u, r, n in discarded]

        domains = Counter(c.domain for c in state["search_queue"])
        ranked = sorted([score_candidate(c, state["active_subqueries"], state["visited_urls"], domains) for c in deduped], key=lambda x: x.score, reverse=True)

        if not ranked:
            event = self._runtime.telemetry.record("source_manager", "No results", verbosity=1, payload_type="queue")
            detail_event = self._runtime.telemetry.record(
                "source_manager",
                "Search produced no new ranked candidates",
                verbosity=3,
                payload_type="web_candidates",
                queries=queries,
                discarded_count=len(discarded),
            )
            return {
                "completed_search_queries": [*state["completed_search_queries"], *queries],
                "discarded_sources": new_discarded,
                "current_candidate": None,
                "current_browser_result": None,
                "latest_evidence": [],
                "progress_score": 0,
                "iteration": state["iteration"] + 1,
                "technical_reason": "no_results",
                "telemetry": self._runtime.telemetry.extend(state["telemetry"], event, detail_event),
            }

        next_c = ranked[0]
        event = self._runtime.telemetry.record("source_manager", "Discovered", verbosity=1, payload_type="queue", url=next_c.url, count=len(ranked))
        detail_event = self._runtime.telemetry.record(
            "source_manager",
            "Ranked search candidates",
            verbosity=3,
            payload_type="web_candidates",
            queries=queries,
            top_candidates=summarize_search_candidates(ranked),
            discarded_count=len(discarded),
        )
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
            "telemetry": self._runtime.telemetry.extend(state["telemetry"], event, detail_event),
        }
