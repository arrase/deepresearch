"""Extractor node implementation.

Iterates over every source in ``current_batch`` and makes one LLM extraction
call per source. Drafts from all sources are collected into a single
``extracted_evidence_buffer``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import sanitize_source_title, select_relevant_chunks, split_text
from ..state import EvidenceDraft, ResearchState, ResearchTopic, SearchCandidate
from .base import accumulate_usage_totals, log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class ExtractorNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _build_local_sources(self, candidate: SearchCandidate, topic: ResearchTopic) -> list[str]:
        terms = topic.search_terms or [topic.question]
        max_passes = self._runtime.config.runtime.max_extraction_passes_per_source
        max_chars = min(
            self._runtime.config.search.max_raw_content_chars,
            self._runtime.config.runtime.extraction_max_chars_per_pass,
        )
        selected_chunks = select_relevant_chunks(split_text(candidate.raw_content), terms, max(8, max_passes * 4))
        deduped_chunks: list[str] = []
        seen: set[str] = set()
        for chunk in selected_chunks:
            normalized = chunk.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            deduped_chunks.append(normalized)

        passes: list[str] = []
        current: list[str] = []
        current_chars = 0
        for chunk in deduped_chunks:
            chunk_chars = len(chunk)
            if chunk_chars >= max_chars:
                if current:
                    passes.append("\n\n".join(current)[:max_chars])
                    current = []
                    current_chars = 0
                passes.append(chunk[:max_chars])
                if len(passes) >= max_passes:
                    break
                continue
            projected = current_chars + chunk_chars + (2 if current else 0)
            if current and projected > max_chars:
                passes.append("\n\n".join(current)[:max_chars])
                if len(passes) >= max_passes:
                    break
                current = [chunk]
                current_chars = chunk_chars
                continue
            current.append(chunk)
            current_chars = projected
        if current and len(passes) < max_passes:
            passes.append("\n\n".join(current)[:max_chars])
        if not passes:
            passes.append(candidate.raw_content[:max_chars])
        return passes[:max_passes]

    def _extract_candidate(
        self,
        state: ResearchState,
        topic: ResearchTopic,
        candidate: SearchCandidate,
    ) -> tuple[list[EvidenceDraft], dict[str, int]]:
        if not candidate.raw_content.strip():
            return [], {}

        all_drafts: list[EvidenceDraft] = []
        total_usage: dict[str, int] = {}
        for local_source in self._build_local_sources(candidate, topic):
            context = self._runtime.context_manager.extractor_context(state, topic, local_source)
            payload, usage = self._runtime.llm_workers.extract_evidence_with_usage(context)
            total_usage = accumulate_usage_totals(total_usage, usage)
            all_drafts.extend(
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
            )
        log_runtime_event(
            self._runtime,
            "[extractor] Extraction complete",
            verbosity=1,
            url=candidate.url,
            count=len(all_drafts),
            passes=len(self._build_local_sources(candidate, topic)),
            **total_usage,
        )
        return all_drafts, total_usage

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
            drafts, usage = self._extract_candidate(state, topic, candidate)
            total_usage = accumulate_usage_totals(total_usage, usage)
            all_drafts.extend(drafts)

        llm_usage = update_stage_llm_usage(state.get("llm_usage", {}), "extractor", total_usage)
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
