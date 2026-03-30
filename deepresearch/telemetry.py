"""Auditable telemetry and checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from .state import FinalReport, ResearchState, TelemetryEvent


class TelemetryRecorder:
    def __init__(self, *, artifacts_dir: Path, logs_dir: Path, echo_to_console: bool = False) -> None:
        self._artifacts_dir = artifacts_dir
        self._logs_dir = logs_dir
        self._echo_to_console = echo_to_console
        self._event_log = self._logs_dir / "research_events.jsonl"
        self._checkpoint_dir = self._artifacts_dir / "checkpoints"
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._event_log.parent.mkdir(parents=True, exist_ok=True)

    def record(self, stage: str, message: str, **payload: Any) -> TelemetryEvent:
        event = TelemetryEvent(stage=stage, message=message, payload=payload)
        with self._event_log.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")
        if self._echo_to_console:
            payload_text = ""
            if payload:
                payload_text = " | " + ", ".join(f"{key}={value}" for key, value in payload.items())
            print(f"[{event.timestamp}] {stage}: {message}{payload_text}", file=sys.stderr, flush=True)
        return event

    def checkpoint(self, state: ResearchState, *, label: str) -> Path:
        safe_label = label.replace(" ", "_")
        target = self._checkpoint_dir / f"{state['iteration']:02d}_{safe_label}.json"
        serializable = dict(state)
        serializable["working_dossier"] = state["working_dossier"].model_dump()
        serializable["context_window_config"] = state["context_window_config"].model_dump()
        serializable["active_subqueries"] = [item.model_dump() for item in state["active_subqueries"]]
        serializable["resolved_subqueries"] = [item.model_dump() for item in state["resolved_subqueries"]]
        serializable["search_intents"] = [item.model_dump() for item in state["search_intents"]]
        serializable["search_queue"] = [item.model_dump() for item in state["search_queue"]]
        serializable["visited_urls"] = {
            key: value.model_dump()
            for key, value in state["visited_urls"].items()
        }
        serializable["discarded_sources"] = [item.model_dump() for item in state["discarded_sources"]]
        serializable["atomic_evidence"] = [item.model_dump() for item in state["atomic_evidence"]]
        serializable["contradictions"] = [item.model_dump() for item in state["contradictions"]]
        serializable["open_gaps"] = [item.model_dump() for item in state["open_gaps"]]
        serializable["telemetry"] = [item.model_dump() for item in state["telemetry"]]
        if state["final_report"] is not None:
            serializable["final_report"] = state["final_report"].model_dump()
        if state.get("current_candidate") is not None:
            serializable["current_candidate"] = state["current_candidate"].model_dump()
        if state.get("current_browser_result") is not None:
            serializable["current_browser_result"] = state["current_browser_result"].model_dump()
        serializable["latest_evidence"] = [item.model_dump() for item in state.get("latest_evidence", [])]
        target.write_text(json.dumps(serializable, ensure_ascii=True, indent=2), encoding="utf-8")
        return target

    def write_markdown_report(self, report: FinalReport, *, label: str = "final_report") -> Path:
        safe_label = label.replace(" ", "_")
        target = self._artifacts_dir / f"{safe_label}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.markdown_report, encoding="utf-8")
        return target
