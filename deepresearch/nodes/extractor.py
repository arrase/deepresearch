"""Extractor node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import select_relevant_chunks, split_text, summarize_evidence
from ..state import BrowserPageStatus, EvidenceDraft, ResearchState
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


_MAX_EXTRACTION_CONTENT = 4000


class ExtractorNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    @traceable(name="extractor-node")
    @log_node_activity("extractor", "Extracting from: {query}")
    def __call__(self, state: ResearchState) -> dict:
        browser_result = state.get("current_browser_result")
        active_topic_id = state.get("active_topic_id")
        topic = next((item for item in state["plan"] if item.id == active_topic_id), None)
        if browser_result is None or topic is None or browser_result.status not in {
            BrowserPageStatus.USEFUL,
            BrowserPageStatus.PARTIAL,
        }:
            return {"extracted_evidence_buffer": []}

        terms = topic.search_terms or [topic.question]
        chunks = select_relevant_chunks(split_text(browser_result.content), terms, 8)
        max_chars = min(self._runtime.config.browser.max_content_chars, _MAX_EXTRACTION_CONTENT)
        local_source = "\n\n".join(chunks)[:max_chars]
        context = self._runtime.context_manager.extractor_context(state, topic, local_source)
        payload, usage = self._runtime.llm_workers.extract_evidence_with_usage(context)

        drafts = [
            EvidenceDraft(
                topic_id=topic.id,
                source_url=browser_result.final_url or browser_result.url,
                source_title=browser_result.title or "Unknown Source",
                claim=item.claim,
                quotation=item.quotation,
                locator=item.citation_locator,
                summary=item.summary,
                extractor_output_tokens=usage.get("output_tokens", 0),
                extractor_input_tokens=usage.get("input_tokens", 0),
                extraction_confidence=item.confidence,
                relevance_score=item.relevance_score,
                caveats=item.caveats,
            )
            for item in payload.evidences
        ]
        llm_usage = {**state.get("llm_usage", {}), "extractor": usage}
        log_runtime_event(
            self._runtime,
            "[extractor] Extraction complete",
            verbosity=1,
            url=browser_result.url,
            count=len(drafts),
            **usage,
        )
        log_runtime_event(
            self._runtime,
            "[extractor] Extracted evidence drafts",
            verbosity=3,
            topic_id=topic.id,
            extracted_evidence=summarize_evidence([]),
        )
        return {
            "extracted_evidence_buffer": drafts,
            "llm_usage": llm_usage,
        }
