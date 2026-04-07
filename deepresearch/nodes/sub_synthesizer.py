"""Sub-synthesiser node: produces a ChapterDraft from per-chapter evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..state import (
    ChapterDraft,
    ConfidenceLevel,
    ResearchState,
    ResearchTopic,
    TopicStatus,
)
from .base import log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class SubSynthesizerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _get_chapter(self, state: ResearchState) -> ResearchTopic | None:
        chapter_id = state.get("current_chapter_id")
        if not chapter_id:
            return None
        return next((t for t in state["plan"] if t.id == chapter_id and t.depth == 0), None)

    def _fallback_draft(self, chapter: ResearchTopic) -> ChapterDraft:
        chapter_id = chapter.chapter_id or chapter.id
        return ChapterDraft(
            chapter_id=chapter_id,
            title=chapter.question,
            executive_summary="No curated evidence was retained for this chapter.",
            confidence=ConfidenceLevel.LOW,
            limitations=["No evidence was accepted."],
        )

    @traceable(name="sub-synthesizer-node")
    @log_node_activity("sub_synthesizer", "Synthesising chapter: {query}")
    def __call__(self, state: ResearchState) -> dict:
        chapter = self._get_chapter(state)
        if chapter is None:
            return {}

        chapter_id = chapter.chapter_id or chapter.id
        context = self._runtime.context_manager.sub_synthesizer_context(state, chapter)

        if not context.evidentiary:
            draft = self._fallback_draft(chapter)
            usage: dict[str, int] = {}
        else:
            try:
                draft, usage = self._runtime.llm_workers.sub_synthesize_with_usage(context, chapter_id)
            except (ValueError, KeyError, OSError):
                draft = self._fallback_draft(chapter)
                usage = {}

        # Mark chapter topics as COMPLETED
        plan = [
            t.model_copy(update={"status": TopicStatus.COMPLETED})
            if t.chapter_id == chapter_id
            else t
            for t in state["plan"]
        ]

        completed_ids = list(state.get("completed_chapter_ids") or [])
        if chapter_id not in completed_ids:
            completed_ids.append(chapter_id)

        # Context flush: add chapter_id to flushed set
        flushed = list(state.get("flushed_chapter_ids") or [])
        if chapter_id not in flushed:
            flushed.append(chapter_id)

        llm_usage = update_stage_llm_usage(state.get("llm_usage", {}), "sub_synthesizer", usage)
        log_runtime_event(
            self._runtime,
            "[sub_synthesizer] Chapter draft produced",
            verbosity=1,
            chapter_id=chapter_id,
            evidence_count=len(context.evidentiary),
            sections=len(draft.sections),
            **usage,
        )
        return {
            "chapter_drafts": [*state.get("chapter_drafts", []), draft],
            "completed_chapter_ids": completed_ids,
            "flushed_chapter_ids": flushed,
            "plan": plan,
            "current_chapter_id": None,
            "audit_approved": False,
            "llm_usage": llm_usage,
        }
