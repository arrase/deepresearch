"""Extractor node implementation."""

from __future__ import annotations

from typing import Any

from ..core.utils import select_relevant_chunks, split_text
from ..state import AtomicEvidence, BrowserPageStatus, ResearchState
from .base import record_telemetry


class ExtractorNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("extractor", "Extracting from: {query}")
    def __call__(self, state: ResearchState) -> dict:
        browser_res = state.get("current_browser_result")
        candidate = state.get("current_candidate")
        if not (browser_res and candidate) or browser_res.status not in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
            return {"latest_evidence": []}

        targets = candidate.subquery_ids or [state["active_subqueries"][0].id]
        terms = [t for sq in state["active_subqueries"] if sq.id in targets for t in (sq.search_terms or [sq.question])]
        
        chunks = select_relevant_chunks(split_text(browser_res.content), terms, 10)
        local_source = "\n\n".join(chunks)[:self._runtime.config.browser.max_content_chars]
        
        context = self._runtime.context_manager.extractor_context(state, targets, local_source)
        payload = self._runtime.llm_workers.extract_evidence(context)
        
        latest = [
            AtomicEvidence(
                subquery_id=targets[0],
                source_url=browser_res.final_url or browser_res.url,
                source_title=candidate.title or browser_res.title or "Unknown Source",
                **item.model_dump()
            ) for item in payload.evidences
        ]
        
        event = self._runtime.telemetry.record("extractor", "Extraction complete", url=browser_res.url, count=len(latest))
        return {"latest_evidence": latest, "telemetry": [*state["telemetry"], event]}
