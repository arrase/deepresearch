"""Global synthesiser node: assembles chapter drafts into a final report."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import build_report_sources
from ..state import ConfidenceLevel, FinalReport, ResearchState, StopReason
from .base import log_node_activity, log_runtime_event, update_stage_llm_usage

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class GlobalSynthesizerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _fallback_report(self, state: ResearchState) -> FinalReport:
        gap_lines = [gap.description for gap in state["open_gaps"][:5]]
        markdown = "\n\n".join(
            [
                "# Research Report",
                "## Outcome",
                "No chapter drafts were produced for this query.",
                "## Limitations",
                "- The system could not generate chapter-level syntheses.",
                "## Open Gaps",
                *([f"- {gap}" for gap in gap_lines] or ["- No explicit gaps were recorded."]),
            ]
        )
        return FinalReport(
            query=state["query"],
            executive_answer="No chapter drafts were produced.",
            key_findings=["The system could not produce chapter-level syntheses."],
            confidence=ConfidenceLevel.LOW,
            reservations=["The result is based on failed or incomplete research."],
            open_gaps=gap_lines,
            cited_sources=[],
            evidence_ids=[],
            markdown_report=markdown,
        )

    @traceable(name="global-synthesizer-node")
    @log_node_activity("global_synthesizer", "Assembling final report: {query}")
    def __call__(self, state: ResearchState) -> dict:
        drafts = state.get("chapter_drafts") or []
        if not drafts:
            report = self._fallback_report(state)
            stop_reason = state.get("stop_reason") or StopReason.PLAN_COMPLETED.value
            return {"final_report": report, "stop_reason": stop_reason}

        context = self._runtime.context_manager.global_synthesizer_context(state)
        synthesis_budget = self._runtime.context_manager.synthesis_budget(state)

        try:
            report, usage = self._runtime.llm_workers.global_synthesize_with_usage(context, query=state["query"])
        except (ValueError, KeyError, OSError):
            report = self._fallback_report(state)
            usage = {}

        # Set the stop reason if no early-termination reason was set
        stop_reason = state.get("stop_reason") or StopReason.PLAN_COMPLETED.value

        report.stop_reason = stop_reason
        report.context_window_tokens = synthesis_budget.context_window_tokens
        report.reserved_output_tokens = synthesis_budget.reserved_output_tokens
        report.prompt_tokens = synthesis_budget.base_prompt_tokens
        report.evidence_tokens = synthesis_budget.selected_evidence_tokens
        report.available_prompt_tokens = synthesis_budget.available_prompt_tokens
        report.llm_usage = usage

        # Rebuild cited_sources from actual curated evidence
        report.cited_sources = build_report_sources(state["curated_evidence"])
        report.evidence_ids = [e.evidence_id for e in state["curated_evidence"]]

        # Set the stop reason if no early-termination reason was set
        stop_reason = state.get("stop_reason") or StopReason.PLAN_COMPLETED.value

        llm_usage = update_stage_llm_usage(state.get("llm_usage", {}), "global_synthesizer", usage)
        log_runtime_event(
            self._runtime,
            "[global_synthesizer] Report generated",
            verbosity=1,
            chapters=len(drafts),
            sources=len(report.cited_sources),
            stop_reason=report.stop_reason,
            **usage,
        )
        return {"final_report": report, "llm_usage": llm_usage, "stop_reason": stop_reason}
