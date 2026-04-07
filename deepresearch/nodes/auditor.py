"""Auditor node: devil's-advocate review of a chapter's evidence."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..state import ResearchState, ResearchTopic, TopicStatus
from .base import log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class AuditorNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _get_chapter(self, state: ResearchState) -> ResearchTopic | None:
        chapter_id = state.get("current_chapter_id")
        if not chapter_id:
            return None
        return next((t for t in state["plan"] if t.id == chapter_id and t.depth == 0), None)

    @traceable(name="auditor-node")
    @log_node_activity("auditor", "Auditing chapter evidence: {query}")
    def __call__(self, state: ResearchState) -> dict:
        chapter = self._get_chapter(state)
        if chapter is None:
            return {"audit_approved": True}

        chapter_id = chapter.chapter_id or chapter.id
        audit_attempts = dict(state.get("topic_audit_attempts") or {})
        current_attempts = audit_attempts.get(chapter_id, 0)
        max_rejections = self._runtime.config.runtime.max_audit_rejections

        # Auto-approve if max rejections reached
        if current_attempts >= max_rejections:
            log_runtime_event(
                self._runtime,
                "[auditor] Auto-approved (max rejections reached)",
                verbosity=1,
                chapter_id=chapter_id,
                attempts=current_attempts,
            )
            return {"audit_approved": True, "topic_audit_attempts": audit_attempts}

        context = self._runtime.context_manager.auditor_context(state, chapter)
        payload, usage = self._runtime.llm_workers.audit_evidence_with_usage(context)
        audit_attempts[chapter_id] = current_attempts + 1

        llm_usage = update_stage_llm_usage(state.get("llm_usage", {}), "auditor", usage)

        if payload.approved:
            log_runtime_event(
                self._runtime,
                "[auditor] Chapter approved",
                verbosity=1,
                chapter_id=chapter_id,
                rationale=payload.rationale[:200],
                **usage,
            )
            return {
                "audit_approved": True,
                "topic_audit_attempts": audit_attempts,
                "llm_usage": llm_usage,
            }

        # Rejected: create child topics from suggestions
        max_depth = self._runtime.config.runtime.max_topic_depth
        known_ids = {t.id for t in state["plan"]}
        new_topics: list[ResearchTopic] = []
        for suggested in payload.suggested_topics:
            if suggested.id in known_ids:
                continue
            suggested.parent_id = chapter_id
            suggested.chapter_id = chapter_id
            suggested.depth = min(suggested.depth if suggested.depth > 0 else 1, max_depth)
            if suggested.depth > max_depth:
                continue
            suggested.status = TopicStatus.PENDING
            new_topics.append(suggested)

        plan = [*state["plan"], *new_topics]

        # Record open gaps from objections
        from ..state import Gap, GapSeverity
        new_gaps = [
            Gap(topic_id=chapter_id, description=obj, severity=GapSeverity.HIGH)
            for obj in payload.objections
        ]

        log_runtime_event(
            self._runtime,
            "[auditor] Chapter rejected",
            verbosity=1,
            chapter_id=chapter_id,
            objections=len(payload.objections),
            new_topics=len(new_topics),
            rationale=payload.rationale[:200],
            **usage,
        )
        return {
            "audit_approved": False,
            "topic_audit_attempts": audit_attempts,
            "plan": plan,
            "open_gaps": [*state["open_gaps"], *new_gaps],
            "llm_usage": llm_usage,
        }
