"""Browser node implementation."""

from __future__ import annotations

from typing import Any

from ..state import BrowserPageStatus, BrowserResult, DiscardedSource, ResearchState, SourceDiscardReason, SourceVisit
from .base import record_telemetry


class BrowserNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("browser", "Navigating to: {query}")
    def __call__(self, state: ResearchState) -> dict:
        candidate = state.get("current_candidate")
        if candidate is None:
            result = BrowserResult(
                url="",
                status=BrowserPageStatus.TERMINAL_ERROR,
                error="No actionable candidate is available",
            )
            event = self._runtime.telemetry.record(
                "browser",
                "Skipping navigation because there is no current candidate",
            )
            return {
                "current_browser_result": result,
                "telemetry": [*state["telemetry"], event],
            }

        result = self._runtime.browser.fetch(candidate.url)
        visited = dict(state["visited_urls"])
        resolved_title = candidate.title or result.title
        visited[result.url] = SourceVisit(
            url=result.url,
            final_url=result.final_url,
            title=resolved_title,
            status=result.status,
            content_excerpt=result.excerpt,
            error=result.error,
            candidate_subquery_ids=candidate.subquery_ids,
            diagnostics=result.diagnostics,
        )
        discarded_sources = [*state["discarded_sources"]]
        if result.status not in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
            reason_map = {
                BrowserPageStatus.BLOCKED: SourceDiscardReason.BLOCKED,
                BrowserPageStatus.EMPTY: SourceDiscardReason.EMPTY,
                BrowserPageStatus.RECOVERABLE_ERROR: SourceDiscardReason.TECHNICAL_ERROR,
                BrowserPageStatus.TERMINAL_ERROR: SourceDiscardReason.TECHNICAL_ERROR,
            }
            discarded_sources.append(
                DiscardedSource(
                    url=candidate.url,
                    reason=reason_map.get(result.status, SourceDiscardReason.LOW_VALUE),
                    note=result.error or result.status.value,
                )
            )
        event = self._runtime.telemetry.record(
            "browser",
            "Navigation completed",
            url=candidate.url,
            status=result.status.value,
        )
        return {
            "visited_urls": visited,
            "discarded_sources": discarded_sources,
            "current_browser_result": result,
            "telemetry": [*state["telemetry"], event],
        }
