"""Extractor node implementation.

Iterates over every source in ``current_batch`` and makes one LLM extraction
call per source. Drafts from all sources are collected into a single
``extracted_evidence_buffer``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import sanitize_source_title, select_relevant_chunks, split_text
from ..state import EvidenceDraft, ResearchState
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
        candidates = state["current_batch"]
        active_topic_id = state.get("active_topic_id")
        topic = next((item for item in state["plan"] if item.id == active_topic_id), None)
        if topic is None:
            return {"extracted_evidence_buffer": []}

        all_drafts: list[EvidenceDraft] = []
        total_usage: dict[str, int] = {}

        for candidate in candidates:
            if not candidate.raw_content.strip():
                continue

            terms = topic.search_terms or [topic.question]
            chunks = select_relevant_chunks(split_text(candidate.raw_content), terms, 8)
            max_chars = min(self._runtime.config.search.max_raw_content_chars, _MAX_EXTRACTION_CONTENT)
            local_source = "\n\n".join(chunks)[:max_chars]
            context = self._runtime.context_manager.extractor_context(state, topic, local_source)
            payload, usage = self._runtime.llm_workers.extract_evidence_with_usage(context)

            for key in ("input_tokens", "output_tokens", "total_tokens"):
                total_usage[key] = total_usage.get(key, 0) + usage.get(key, 0)

            drafts = [
                EvidenceDraft(
                    topic_id=topic.id,
                    source_url=candidate.normalized_url or candidate.url,
                    source_title=sanitize_source_title(
                        candidate.title,
                        candidate.normalized_url or candidate.url,
                    )
                    or "Unknown Source",
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
            all_drafts.extend(drafts)
            log_runtime_event(
                self._runtime,
                "[extractor] Extraction complete",
                verbosity=1,
                url=candidate.url,
                count=len(drafts),
                **usage,
            )

        llm_usage = {**state.get("llm_usage", {}), "extractor": total_usage}
        log_runtime_event(
            self._runtime,
            "[extractor] Batch extraction finished",
            verbosity=2,
            total_drafts=len(all_drafts),
            sources_processed=sum(1 for candidate in candidates if candidate.raw_content.strip()),
        )
        return {
            "extracted_evidence_buffer": all_drafts,
            "llm_usage": llm_usage,
        }
