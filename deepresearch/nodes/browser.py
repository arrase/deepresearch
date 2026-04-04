"""Browser node implementation.

Processes the full ``current_batch`` of candidates.  For each candidate the
node first checks if usable ``raw_content`` was already provided by Tavily.
When the raw text is long enough the browser fetch is skipped entirely,
saving a Docker round-trip.  Falls back to Lightpanda only when necessary.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import classify_browser_payload, sanitize_source_title, summarize_source_visit
from ..state import BrowserPageStatus, DiscardedSource, ResearchState, SourceDiscardReason, SourceRecord, SourceVisit
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime
    from ..state import SearchCandidate

logger = logging.getLogger(__name__)

_REASON_MAP = {
    BrowserPageStatus.BLOCKED: SourceDiscardReason.BLOCKED,
    BrowserPageStatus.EMPTY: SourceDiscardReason.EMPTY,
    BrowserPageStatus.RECOVERABLE_ERROR: SourceDiscardReason.TECHNICAL_ERROR,
    BrowserPageStatus.TERMINAL_ERROR: SourceDiscardReason.TECHNICAL_ERROR,
}


class BrowserNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _visit_from_raw_content(self, candidate: SearchCandidate) -> SourceVisit | None:
        """Build a SourceVisit from Tavily raw_content if it is long enough."""
        min_chars = self._runtime.config.browser.min_useful_chars
        raw = (candidate.raw_content or "").strip()
        if len(raw) < min_chars:
            return None
        status = classify_browser_payload(
            content=raw,
            error=None,
            exit_code=0,
            min_partial_chars=self._runtime.config.browser.min_partial_chars,
            min_useful_chars=min_chars,
        )
        if status not in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
            return None
        return SourceVisit(
            url=candidate.url,
            final_url=candidate.url,
            title=sanitize_source_title(candidate.title, candidate.url) or "Unknown Title",
            status=status,
            content=raw,
            excerpt=raw[:300],
            topic_ids=list(candidate.topic_ids),
        )

    def _visit_via_browser(self, candidate: SearchCandidate) -> SourceVisit:
        """Fetch page content through the Lightpanda headless browser."""
        try:
            result = self._runtime.browser.fetch(candidate.url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[browser] Fetch failed for %s: %s", candidate.url, exc)
            return SourceVisit(
                url=candidate.url,
                status=BrowserPageStatus.TERMINAL_ERROR,
                error=str(exc),
                topic_ids=list(candidate.topic_ids),
            )
        result.topic_ids = list(dict.fromkeys([*result.topic_ids, *candidate.topic_ids]))
        result.title = (
            sanitize_source_title(result.title)
            or sanitize_source_title(candidate.title, candidate.url)
            or "Unknown Title"
        )
        return result

    def _process_candidate(self, candidate: SearchCandidate) -> SourceVisit:
        """Return a SourceVisit for *candidate*, preferring raw_content."""
        visit = self._visit_from_raw_content(candidate)
        if visit is not None:
            log_runtime_event(
                self._runtime,
                "[browser] Using Tavily raw_content (skipping headless browser)",
                verbosity=1,
                url=candidate.url,
            )
            return visit
        return self._visit_via_browser(candidate)

    @traceable(name="browser-node")
    @log_node_activity("browser", "Navigating to: {query}")
    def __call__(self, state: ResearchState) -> dict:
        if not state["current_batch"]:
            return {"browser_results": [], "useful_source_in_cycle": False}

        visited = dict(state["visited_urls"])
        discarded_sources = list(state["discarded_sources"])
        results: list[SourceVisit] = []
        any_useful = False
        any_browser_error = False

        for candidate in state["current_batch"]:
            result = self._process_candidate(candidate)
            results.append(result)

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

            if result.status in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
                any_useful = True
            else:
                discarded_sources.append(
                    DiscardedSource(
                        url=candidate.url,
                        reason=_REASON_MAP.get(result.status, SourceDiscardReason.LOW_VALUE),
                        note=result.error or result.status.value,
                    )
                )
                if result.status in {BrowserPageStatus.RECOVERABLE_ERROR, BrowserPageStatus.TERMINAL_ERROR}:
                    any_browser_error = True

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
            "browser_results": results,
            "useful_source_in_cycle": any_useful,
            "technical_reason": "browser_error" if (any_browser_error and not any_useful) else None,
        }
