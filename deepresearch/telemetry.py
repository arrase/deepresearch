"""Auditable telemetry and checkpoints."""

from __future__ import annotations

import sys
from typing import Any

from .state import TelemetryEvent


class TelemetryRecorder:
    def __init__(self, *, echo_to_console: bool = False) -> None:
        self._echo_to_console = echo_to_console

    def record(self, stage: str, message: str, **payload: Any) -> TelemetryEvent:
        event = TelemetryEvent(stage=stage, message=message, payload=payload)
        if self._echo_to_console:
            payload_text = ""
            if payload:
                payload_text = " | " + ", ".join(f"{key}={value}" for key, value in payload.items())
            print(f"[{event.timestamp}] {stage}: {message}{payload_text}", file=sys.stderr, flush=True)
        return event
