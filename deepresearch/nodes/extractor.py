"""Extractor node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.utils import rank_subqueries_for_source, select_relevant_chunks, split_text, summarize_evidence
from ..state import AtomicEvidence, BrowserPageStatus, ResearchState
from .base import consume_llm_telemetry_events, record_telemetry

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime

_MAX_EXTRACTION_CONTENT = 4000


class ExtractorNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    @record_telemetry("extractor", "Extracting from: {query}")
    def __call__(self, state: ResearchState) -> dict:
        browser_res = state.get("current_browser_result")
        candidate = state.get("current_candidate")
        if not (browser_res and candidate) or browser_res.status not in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
            return {"latest_evidence": []}

        source_text = "\n".join(
            part
            for part in [
                candidate.title,
                candidate.snippet,
                browser_res.title,
                browser_res.excerpt,
                browser_res.content[:_MAX_EXTRACTION_CONTENT],
            ]
            if part
        )
        targets = rank_subqueries_for_source(
            state["active_subqueries"],
            text=source_text,
            candidate_subquery_ids=candidate.subquery_ids or browser_res.candidate_subquery_ids,
        )
        if not targets:
            return {"latest_evidence": []}
        terms = [t for sq in state["active_subqueries"] if sq.id in targets for t in (sq.search_terms or [sq.question])]

        chunks = select_relevant_chunks(split_text(browser_res.content), terms, 10)
        local_source = "\n\n".join(chunks)[:self._runtime.config.browser.max_content_chars]

        context = self._runtime.context_manager.extractor_context(state, targets, local_source)
        payload, usage = self._runtime.llm_workers.extract_evidence_with_usage(context)
        llm_events = consume_llm_telemetry_events(self._runtime)

        latest = []
        target_subqueries = [sq for sq in state["active_subqueries"] if sq.id in targets]
        for item in payload.evidences:
            best_id = targets[0]
            if len(targets) > 1:
                match_ids = rank_subqueries_for_source(
                    target_subqueries,
                    text=f"{item.claim} {item.summary}",
                    candidate_subquery_ids=targets,
                    limit=1,
                )
                if match_ids:
                    best_id = match_ids[0]
            latest.append(AtomicEvidence(
                subquery_id=best_id,
                source_url=browser_res.final_url or browser_res.url,
                source_title=candidate.title or browser_res.title or "Unknown Source",
                **item.model_dump()
            ))
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
