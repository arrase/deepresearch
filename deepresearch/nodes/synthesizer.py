"""Synthesizer node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import build_report_sources
from ..state import ConfidenceLevel, FinalReport, ResearchState
from .base import log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..context_manager import NodeContext
    from ..runtime import ResearchRuntime


class SynthesizerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _estimate_confidence(self, state: ResearchState) -> ConfidenceLevel:
        if not state["curated_evidence"]:
            return ConfidenceLevel.LOW
        gap_count = len(state["open_gaps"])
        contradiction_count = len(state["contradictions"])
        distinct_domains = {
            source.url
            for evidence in state["curated_evidence"]
            for source in evidence.sources
        }
        if contradiction_count or gap_count > max(2, len(state["plan"])):
            return ConfidenceLevel.LOW
        target_total = sum(topic.evidence_target for topic in state["plan"]) or 1
        if len(state["curated_evidence"]) >= target_total and len(distinct_domains) >= max(2, len(state["plan"])):
            return ConfidenceLevel.HIGH
        return ConfidenceLevel.MEDIUM

    def _fallback_report(self, state: ResearchState) -> FinalReport:
        gap_lines = [gap.description for gap in state["open_gaps"][:5]]
        markdown = "\n\n".join(
            [
                "# Research Report",
                "## Outcome",
                "No curated evidence was accepted for the query.",
                "## Limitations",
                "- The system could not validate any source strongly enough to retain evidence.",
                "- The report is partial and should be treated as an absence-of-evidence summary.",
                "## Open Gaps",
                *([f"- {gap}" for gap in gap_lines] or ["- No explicit gaps were recorded."]),
            ]
        )
        return FinalReport(
            query=state["query"],
            executive_answer="No curated evidence was found.",
            key_findings=["The system could not retain any evidence for the query."],
            confidence=ConfidenceLevel.LOW,
            reservations=["The result is based on failed or non-useful research cycles."],
            open_gaps=gap_lines,
            cited_sources=[],
            evidence_ids=[],
            markdown_report=markdown,
        )

    def _failed_report(self, state: ResearchState) -> FinalReport:
        return FinalReport(
            query=state["query"],
            executive_answer="Synthesis failed.",
            key_findings=["Error in synthesis."],
            confidence=ConfidenceLevel.LOW,
            evidence_ids=[item.evidence_id for item in state["curated_evidence"]],
            cited_sources=build_report_sources(state["curated_evidence"]),
            markdown_report="# Research Report\n\nSynthesis failed.",
        )

    def _generate_report(
        self,
        state: ResearchState,
        context: NodeContext,
    ) -> tuple[FinalReport, dict[str, int]]:
        if not state["curated_evidence"]:
            return self._fallback_report(state), {}

        try:
            return self._runtime.llm_workers.synthesize_report_with_usage(context, query=state["query"])
        except (ValueError, KeyError, OSError):
            return self._failed_report(state), {}

    @traceable(name="synthesizer-node")
    @log_node_activity("synthesizer", "Synthesizing report: {query}")
    def __call__(self, state: ResearchState) -> dict:
        context = self._runtime.context_manager.synthesizer_context(state)
        synthesis_budget = state["synthesis_budget"] or self._runtime.context_manager.synthesis_budget(state)
        report, usage = self._generate_report(state, context)
        report.stop_reason = state.get("stop_reason")
        report.context_window_tokens = synthesis_budget.context_window_tokens
        report.reserved_output_tokens = synthesis_budget.reserved_output_tokens
        report.prompt_tokens = synthesis_budget.base_prompt_tokens
        report.evidence_tokens = synthesis_budget.selected_evidence_tokens
        report.available_prompt_tokens = synthesis_budget.available_prompt_tokens
        report.confidence = self._estimate_confidence(state)
        if not report.open_gaps:
            report.open_gaps = [gap.description for gap in state["open_gaps"][:6]]
        report.llm_usage = usage
        llm_usage = update_stage_llm_usage(state.get("llm_usage", {}), "synthesizer", usage)
        log_runtime_event(
            self._runtime,
            "[synthesizer] Report generated",
            verbosity=1,
            sources=len(report.cited_sources),
            stop_reason=report.stop_reason,
            **usage,
        )
        return {"final_report": report, "llm_usage": llm_usage}
