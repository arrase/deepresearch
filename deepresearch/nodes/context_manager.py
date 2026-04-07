"""Curator node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import compute_topic_coverages, curate_evidence, summarize_evidence, update_working_dossier
from ..state import DiscardedSource, ResearchState, SourceDiscardReason
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class ContextManagerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _discard_no_evidence_sources(self, state: ResearchState) -> list[DiscardedSource]:
        discarded = list(state["discarded_sources"])
        if state["current_batch"] and not state["extracted_evidence_buffer"]:
            discarded.extend(
                DiscardedSource(
                    url=candidate.url,
                    reason=SourceDiscardReason.NO_EVIDENCE,
                    note="No evidence extracted from this source",
                )
                for candidate in state["current_batch"]
            )
        return discarded

    def _prompt_fit_tokens(self, curated: list) -> int:
        return sum(item.prompt_fit_tokens_estimate for item in curated)

    @traceable(name="context-manager-node")
    @log_node_activity("context_manager", "Curating evidence for: {query}")
    def __call__(self, state: ResearchState) -> dict:
        drafts = state["extracted_evidence_buffer"]
        curated, accepted, merged_count, exact_added_tokens = curate_evidence(
            state["curated_evidence"],
            drafts,
            iteration=state["current_iteration"],
            dedup_config=self._runtime.config.dedup,
        )

        # Propagate chapter_id from the topic → evidence mapping
        chapter_id = state.get("current_chapter_id") or ""
        topic_chapter_map: dict[str, str] = {}
        for topic in state["plan"]:
            topic_chapter_map[topic.id] = topic.chapter_id or chapter_id

        for evidence in accepted:
            if not evidence.chapter_id:
                evidence.chapter_id = topic_chapter_map.get(evidence.topic_id, chapter_id)

        # Also backfill chapter_id on existing curated evidence if missing
        for evidence in curated:
            if not evidence.chapter_id:
                evidence.chapter_id = topic_chapter_map.get(evidence.topic_id, chapter_id)

        dossier = update_working_dossier(state["working_dossier"], accepted)
        coverage = compute_topic_coverages(state["plan"], curated, state["topic_attempts"])
        discarded = self._discard_no_evidence_sources(state)

        log_runtime_event(self._runtime, "[context_manager] Curated evidence", verbosity=1, accepted=len(accepted))
        log_runtime_event(
            self._runtime,
            "[context_manager] Evidence curation details",
            verbosity=3,
            accepted_evidence=summarize_evidence(accepted),
            merged_count=merged_count,
        )
        return {
            "curated_evidence": curated,
            "working_dossier": dossier,
            "topic_coverage": coverage,
            "discarded_sources": discarded,
            "accumulated_evidence_tokens_exact": state["accumulated_evidence_tokens_exact"] + exact_added_tokens,
            "accumulated_evidence_tokens_prompt_fit": self._prompt_fit_tokens(curated),
            "new_evidence_in_cycle": len(accepted),
            "merged_evidence_in_cycle": merged_count,
        }
