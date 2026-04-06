"""Topic-level synthesis node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..state import ResearchState, TopicBrief
from .base import accumulate_usage_totals, log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class TopicSynthesizerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _fallback_brief(self, state: ResearchState, topic_id: str, question: str) -> TopicBrief:
        summary = state["working_dossier"].topic_summaries.get(topic_id, "No accepted evidence yet.")
        gap_lines = [gap.description for gap in state["open_gaps"] if gap.topic_id == topic_id][:4]
        markdown = "\n".join(
            [
                f"### Topic\n{question}",
                "### Answer",
                summary,
                "### Uncertainty And Gaps",
                *([f"- {gap}" for gap in gap_lines] or ["- No explicit gaps recorded."]),
            ]
        ).strip()
        return TopicBrief(topic_id=topic_id, question=question, markdown_brief=markdown)

    @traceable(name="topic-synthesizer-node")
    @log_node_activity("topic_synthesizer", "Synthesizing topic briefs: {query}")
    def __call__(self, state: ResearchState) -> dict:
        briefs = dict(state.get("topic_briefs", {}))
        total_usage: dict[str, int] = {}

        for topic in state["plan"]:
            context = self._runtime.context_manager.topic_brief_context(state, topic)
            try:
                brief, usage = self._runtime.llm_workers.synthesize_topic_brief_with_usage(
                    context,
                    query=state["query"],
                    topic=topic,
                )
            except (ValueError, KeyError, OSError):
                brief, usage = self._fallback_brief(state, topic.id, topic.question), {}
            briefs[topic.id] = brief
            total_usage = accumulate_usage_totals(total_usage, usage)

        llm_usage = update_stage_llm_usage(
            state.get("llm_usage", {}),
            "topic_synthesizer",
            total_usage,
            include_empty=False,
        )
        log_runtime_event(
            self._runtime,
            "[topic_synthesizer] Topic briefs generated",
            verbosity=1,
            briefs=len(briefs),
            **total_usage,
        )
        return {"topic_briefs": briefs, "llm_usage": llm_usage}
