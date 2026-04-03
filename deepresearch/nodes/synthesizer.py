"""Synthesizer node implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from langsmith import traceable

from ..core.utils import build_report_sources
from ..state import ConfidenceLevel, FinalReport, ResearchState
from .base import log_node_activity, log_runtime_event

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class SynthesizerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    @traceable(name="synthesizer-node")
    @log_node_activity("synthesizer", "Synthesizing report: {query}")
    def __call__(self, state: ResearchState) -> dict:
        context = self._runtime.context_manager.synthesizer_context(state)
        synthesis_budget = state.get("synthesis_budget") or self._runtime.context_manager.synthesis_budget(state)
        try:
            report, usage = self._runtime.llm_workers.synthesize_report_with_usage(
                context,
                query=state["query"],
            )
        except (ValueError, KeyError, OSError):
            usage = {}
            report = FinalReport(
                query=state["query"],
                executive_answer="Synthesis failed.",
                key_findings=["Error in synthesis."],
                confidence=ConfidenceLevel.LOW,
                evidence_ids=[e.id for e in state["atomic_evidence"]],
                cited_sources=build_report_sources(state["atomic_evidence"]),
            )
        report.stop_reason = state.get("stop_reason") or (
            "sufficient_information" if state.get("is_sufficient") else None
        )
        report.context_window_tokens = _budget_int(synthesis_budget, "context_window_tokens")
        report.reserved_output_tokens = _budget_int(synthesis_budget, "reserved_output_tokens")
        report.prompt_tokens = _budget_int(synthesis_budget, "base_prompt_tokens")
        report.evidence_tokens = _budget_int(synthesis_budget, "selected_evidence_tokens")
        report.available_prompt_tokens = _budget_int(synthesis_budget, "available_prompt_tokens")
        report.llm_usage = usage
        llm_usage = {**state.get("llm_usage", {}), "synthesizer": usage}

        log_runtime_event(
            self._runtime,
            "[synthesizer] Report generated",
            verbosity=1,
            sources=len(report.cited_sources),
            stop_reason=report.stop_reason,
            **usage,
        )
        log_runtime_event(
            self._runtime,
            "[synthesizer] Synthesis budget applied",
            verbosity=3,
            synthesis_budget={k: v for k, v in synthesis_budget.items() if isinstance(v, (int, bool, str))},
            cited_sources=[source.model_dump(mode="json") for source in report.cited_sources[:5]],
        )
        return {
            "final_report": report,
            "llm_usage": llm_usage,
        }


def _budget_int(synthesis_budget: Mapping[str, object], key: str) -> int | None:
    value = synthesis_budget.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Expected integer-like synthesis budget value for {key!r}, got {type(value).__name__}")
