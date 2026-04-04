"""Browser node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import summarize_source_visit
from ..state import BrowserPageStatus, DiscardedSource, ResearchState, SourceDiscardReason, SourceRecord, SourceVisit
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class BrowserNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    @traceable(name="browser-node")
    @log_node_activity("browser", "Navigating to: {query}")
    def __call__(self, state: ResearchState) -> dict:
        if not state["current_batch"]:
            result = SourceVisit(
                url="",
                status=BrowserPageStatus.TERMINAL_ERROR,
                error="No actionable candidate is available",
            )
            return {"current_browser_result": result, "useful_source_in_cycle": False}

        candidate = state["current_batch"][0]
        result = self._runtime.browser.fetch(candidate.url)
        result.topic_ids = list(dict.fromkeys([*result.topic_ids, *candidate.topic_ids]))
        if not result.title:
            result.title = candidate.title or "Unknown Title"

        visited = dict(state["visited_urls"])
        visited[candidate.normalized_url or candidate.url] = SourceRecord(
            url=candidate.normalized_url or candidate.url,
            final_url=result.final_url or result.url,
            title=result.title,
            fetch_status=result.status,
            extracted_chars=len(result.content),
            relevant_chunks=[],
            topic_ids=result.topic_ids,
            last_error=result.error,
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
            "useful_source_in_cycle": result.status in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL},
            "technical_reason": "browser_error"
            if result.status in {BrowserPageStatus.RECOVERABLE_ERROR, BrowserPageStatus.TERMINAL_ERROR}
            else None,
        }
