"""Context manager node implementation."""

from __future__ import annotations

from typing import Any

from ..core.utils import deduplicate_evidence, update_working_dossier
from ..state import DiscardedSource, ResearchState, SourceDiscardReason
from .base import record_telemetry


class ContextManagerNode:
    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @record_telemetry("context_manager", "Integrating evidence for: {query}")
    def __call__(self, state: ResearchState) -> dict:
        browser_result = state.get("current_browser_result")
        latest = state.get("latest_evidence", [])
        
        accepted = deduplicate_evidence(state["atomic_evidence"], latest)
        updated_evidence = [*state["atomic_evidence"], *accepted]
        
        dossier = update_working_dossier(
            state["working_dossier"],
            evidence=accepted,
            source_url=browser_result.url if browser_result else None,
            source_title=browser_result.title if browser_result else None,
        )
        
        discarded = list(state["discarded_sources"])
        if browser_result and not latest:
            discarded.append(DiscardedSource(
                url=browser_result.url,
                reason=SourceDiscardReason.NO_EVIDENCE,
                note="No evidence extracted from this source"
            ))

        event = self._runtime.telemetry.record("context_manager", "Dossier updated", count=len(accepted))
        return {
            "atomic_evidence": updated_evidence,
            "working_dossier": dossier,
            "discarded_sources": discarded,
            "latest_evidence": accepted,
            "telemetry": [*state["telemetry"], event],
        }
