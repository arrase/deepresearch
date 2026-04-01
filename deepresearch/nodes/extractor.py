"""Extractor node implementation."""

from __future__ import annotations

from typing import Any

from ..core.utils import select_relevant_chunks, split_text, summarize_evidence
from ..state import AtomicEvidence, BrowserPageStatus, ResearchState
from .base import consume_llm_telemetry_events, record_telemetry


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
        payload, usage = self._runtime.llm_workers.extract_evidence_with_usage(context)
        llm_events = consume_llm_telemetry_events(self._runtime)
        
        latest = [
            AtomicEvidence(
                subquery_id=targets[0],
                source_url=browser_res.final_url or browser_res.url,
                source_title=candidate.title or browser_res.title or "Unknown Source",
                **item.model_dump()
            ) for item in payload.evidences
        ]
        llm_usage = {**state.get("llm_usage", {}), "extractor": usage}
        
        event = self._runtime.telemetry.record("extractor", "Extraction complete", verbosity=1, payload_type="decision", url=browser_res.url, count=len(latest), **usage)
        detail_event = self._runtime.telemetry.record(
            "extractor",
            "Processed page chunks and extracted evidence",
            verbosity=3,
            payload_type="web_extraction",
            target_subquery_ids=targets,
            selected_chunks=[{"index": index, "preview": chunk} for index, chunk in enumerate(chunks[:6], start=1)],
            extracted_evidence=summarize_evidence(latest),
        )
        return {
            "latest_evidence": latest,
            "llm_usage": llm_usage,
            "telemetry": self._runtime.telemetry.extend(state["telemetry"], *llm_events, event, detail_event),
        }
