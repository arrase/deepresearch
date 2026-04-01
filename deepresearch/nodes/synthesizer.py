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
        context = self._runtime.context_manager.synthesizer_context(state)
        synthesis_budget = state.get("synthesis_budget") or self._runtime.context_manager.synthesis_budget(state)
        try:
            report, usage = self._runtime.llm_workers.synthesize_report_with_usage(
                context,
                query=state["query"]
            )
        except Exception:
            usage = {}
            report = FinalReport(
                query=state["query"], executive_answer="Synthesis failed.",
                key_findings=["Error in synthesis."],
                confidence=ConfidenceLevel.LOW,
                evidence_ids=[e.id for e in state["atomic_evidence"]],
                cited_sources=build_report_sources(state["atomic_evidence"])
            )

        report.stop_reason = state.get("stop_reason") or ("sufficient_information" if state.get("is_sufficient") else None)
        report.context_window_tokens = int(synthesis_budget.get("context_window_tokens")) if synthesis_budget.get("context_window_tokens") is not None else None
        report.reserved_output_tokens = int(synthesis_budget.get("reserved_output_tokens")) if synthesis_budget.get("reserved_output_tokens") is not None else None
        report.prompt_tokens = int(synthesis_budget.get("base_prompt_tokens")) if synthesis_budget.get("base_prompt_tokens") is not None else None
        report.evidence_tokens = int(synthesis_budget.get("selected_evidence_tokens")) if synthesis_budget.get("selected_evidence_tokens") is not None else None
        report.available_prompt_tokens = int(synthesis_budget.get("available_prompt_tokens")) if synthesis_budget.get("available_prompt_tokens") is not None else None
        report.llm_usage = usage
        llm_usage = {**state.get("llm_usage", {}), "synthesizer": usage}

        event = self._runtime.telemetry.record("synthesizer", "Report generated", sources=len(report.cited_sources), stop_reason=report.stop_reason, **usage)
        return {"final_report": report, "llm_usage": llm_usage, "telemetry": [*state["telemetry"], event]}
