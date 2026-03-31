"""Extractor node implementation."""

from __future__ import annotations

from typing import Any

from ..core.utils import select_relevant_chunks, short_excerpt, split_text
from ..state import AtomicEvidence, BrowserPageStatus, ResearchState
from .base import record_telemetry


class ExtractorNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("extractor", "Extracting evidence from: {query}")
    def __call__(self, state: ResearchState) -> dict:
        browser_result = state.get("current_browser_result")
        candidate = state.get("current_candidate")
        if browser_result is None or candidate is None:
            return {"latest_evidence": []}
        if browser_result.status not in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
            return {"latest_evidence": []}

        target_subquery_ids = candidate.subquery_ids or [state["active_subqueries"][0].id]
        query_terms: list[str] = []
        for subquery in state["active_subqueries"]:
            if subquery.id in target_subquery_ids:
                query_terms.extend(subquery.search_terms or [subquery.question])
        chunks = split_text(browser_result.content)
        selected_chunks = select_relevant_chunks(chunks, query_terms=query_terms, max_chunks=10)
        local_source = "\n\n".join(selected_chunks)[: self._runtime.config.browser.max_content_chars]
        
        context = self._runtime.context_manager.extractor_context(
            state,
            target_subquery_ids=target_subquery_ids,
            local_source=local_source,
        )
        payload = self._runtime.llm_workers.extract_evidence(context)
        latest_evidence: list[AtomicEvidence] = []
        primary_subquery_id = target_subquery_ids[0]
        for item in payload.evidences:
            latest_evidence.append(
                AtomicEvidence(
                    subquery_id=primary_subquery_id,
                    source_url=browser_result.final_url or browser_result.url,
                    source_title=candidate.title or browser_result.title or short_excerpt(local_source, 80),
                    summary=item.summary,
                    claim=item.claim,
                    quotation=item.quotation,
                    citation_locator=item.citation_locator,
                    relevance_score=item.relevance_score,
                    confidence=item.confidence,
                    caveats=item.caveats,
                    tags=item.tags,
                )
            )
        
        event = self._runtime.telemetry.record(
            "extractor",
            "Evidence extraction completed",
            url=browser_result.url,
            count=len(latest_evidence),
        )
        return {
            "latest_evidence": latest_evidence,
            "telemetry": [*state["telemetry"], event],
        }
