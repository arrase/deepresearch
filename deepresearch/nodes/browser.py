"""Browser node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import summarize_source_visit
from ..state import BrowserPageStatus, DiscardedSource, ResearchState, SourceDiscardReason, SourceVisit
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class BrowserNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    @traceable(name="browser-node")
    @log_node_activity("browser", "Navigating to: {query}")
    def __call__(self, state: ResearchState) -> dict:
        candidate = state.get("current_candidate")
        if candidate is None:
            result = SourceVisit(
                url="",
                status=BrowserPageStatus.TERMINAL_ERROR,
                error="No actionable candidate is available",
            )
            log_runtime_event(
                self._runtime,
                "[browser] Skipping navigation because there is no current candidate",
                verbosity=1,
            )
            return {
                "current_browser_result": result,
            }

        result = self._runtime.browser.fetch(candidate.url)
        visited = dict(state["visited_urls"])

        # Preserve original candidate subqueries and ensure title
        result.candidate_subquery_ids = candidate.subquery_ids
        if not result.title:
            result.title = candidate.title or "Unknown Title"

        visited[result.url] = result

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

        log_runtime_event(
            self._runtime,
            "[browser] Navigation completed",
            verbosity=1,
            url=candidate.url,
            status=result.status.value,
        )
        log_runtime_event(
            self._runtime,
            "[browser] Processed web page",
            verbosity=3,
            page=summarize_source_visit(result, include_content_preview=True),
        )
        return {
            "visited_urls": visited,
            "discarded_sources": discarded_sources,
            "current_browser_result": result,
            "useful_sources_count": state["useful_sources_count"]
            + (
                1
                if result.status in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}
                else 0
            ),
        }
