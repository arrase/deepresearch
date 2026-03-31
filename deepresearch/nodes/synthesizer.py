"""Synthesizer node implementation."""

from __future__ import annotations

from typing import Any

from ..state import ResearchState, FinalReport, ConfidenceLevel
from ..core.utils import build_report_sources
from .base import record_telemetry


class SynthesizerNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("synthesizer", "Synthesizing report: {query}")
    def __call__(self, state: ResearchState) -> dict:
        try:
            report = self._runtime.llm_workers.synthesize_report(
                self._runtime.context_manager.synthesizer_context(state), 
                query=state["query"]
            )
        except Exception:
            report = FinalReport(
                query=state["query"], executive_answer="Synthesis failed.",
                key_findings=["Error in synthesis."],
                confidence=ConfidenceLevel.LOW,
                evidence_ids=[e.id for e in state["atomic_evidence"]],
                cited_sources=build_report_sources(state["atomic_evidence"])
            )
            
        event = self._runtime.telemetry.record("synthesizer", "Report generated", sources=len(report.cited_sources))
        return {"final_report": report, "telemetry": [*state["telemetry"], event]}
