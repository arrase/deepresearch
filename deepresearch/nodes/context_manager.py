"""Context manager node implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.utils import deduplicate_evidence, summarize_evidence, update_working_dossier
from ..state import DiscardedSource, ResearchState, SourceDiscardReason
from .base import record_telemetry

if TYPE_CHECKING:
    from ..runtime import ResearchRuntime


class ContextManagerNode:
    def __init__(self, runtime: ResearchRuntime) -> None:
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

        updated_state = {
            **state,
            "atomic_evidence": updated_evidence,
            "working_dossier": dossier,
            "discarded_sources": discarded,
        }
        event = self._runtime.telemetry.record("context_manager", "Dossier updated", verbosity=1, payload_type="dossier", count=len(accepted))
        detail_event = self._runtime.telemetry.record(
            "context_manager",
            "Integrated accepted evidence into dossier",
            verbosity=3,
            payload_type="dossier_snapshot",
            accepted_evidence=summarize_evidence(accepted),
            rejected_count=max(0, len(latest) - len(accepted)),
            snapshot=self._runtime.context_manager.debug_state_snapshot(updated_state),
        )
        return {
            "atomic_evidence": updated_evidence,
            "working_dossier": dossier,
            "discarded_sources": discarded,
            "latest_evidence": accepted,
            "urls_visited_since_eval": state.get("urls_visited_since_eval", 0) + 1,
            "progress_score": (
                len(accepted) * self._runtime.config.runtime.weight_new_evidence
                + (1 if browser_result and browser_result.status.value in {"useful", "partial"} else 0) * self._runtime.config.runtime.weight_useful_source
            ),
            "telemetry": self._runtime.telemetry.extend(state["telemetry"], event, detail_event),
        }
