"""Synthesizer node implementation."""

from __future__ import annotations

from typing import Any

from ..core.utils import render_markdown_report
from ..state import ResearchState
from .base import record_telemetry


class SynthesizerNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("synthesizer", "Synthesizing final report for: {query}")
    def __call__(self, state: ResearchState) -> dict:
        context = self._runtime.context_manager.synthesizer_context(state)
        try:
            report = self._runtime.llm_workers.synthesize_report(context, query=state["query"])
        except Exception:  # noqa: BLE001
            report = self._fallback_report(state)
        
        if not report.markdown_report:
            report.markdown_report = render_markdown_report(report)
            
        event = self._runtime.telemetry.record(
            "synthesizer",
            "Final report generated",
            evidence=len(state["atomic_evidence"]),
            sources=len(report.cited_sources),
        )
        return {
            "final_report": report,
            "telemetry": [*state["telemetry"], event],
        }

    def _fallback_report(self, state: ResearchState) -> Any:
        # Import here to avoid circular imports if any
        from ..state import FinalReport, ConfidenceLevel
        from ..core.utils import build_report_sources
        
        return FinalReport(
            query=state["query"],
            executive_answer="Failed to synthesize a full report due to an internal error.",
            key_findings=["Research was completed but synthesis failed."],
            sections=[],
            confidence=ConfidenceLevel.LOW,
            reservations=["The report synthesis stage encountered a terminal error."],
            evidence_ids=[item.id for item in state["atomic_evidence"]],
            cited_sources=build_report_sources(state["atomic_evidence"]),
        )
