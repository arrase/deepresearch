"""Auditable telemetry and checkpoints."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .state import TelemetryEvent


class TelemetryRecorder:
    _MAX_DEPTH = 4
    _ITEMS_BY_VERBOSITY = {0: 0, 1: 6, 2: 12, 3: 20}
    _CHARS_BY_VERBOSITY = {0: 0, 1: 240, 2: 1200, 3: 4000}
    _ECHO_CHARS_BY_VERBOSITY = {1: 220, 2: 420, 3: 800}

    def __init__(self, *, verbosity: int = 0) -> None:
        self._verbosity = verbosity

    @property
    def verbosity(self) -> int:
        return self._verbosity

    def enabled_for(self, verbosity: int) -> bool:
        return self._verbosity >= verbosity

    def record(
        self,
        stage: str,
        message: str,
        *,
        verbosity: int = 1,
        payload_type: str = "generic",
        **payload: Any,
    ) -> TelemetryEvent | None:
        if not self.enabled_for(verbosity):
            return None

        event = TelemetryEvent(
            stage=stage,
            message=message,
            verbosity=verbosity,
            payload_type=payload_type,
            payload=self._normalize_value(payload, depth=0),
        )
        self._echo(event)
        return event

    def extend(
        self,
        existing: Sequence[TelemetryEvent],
        *events: TelemetryEvent | None,
    ) -> list[TelemetryEvent]:
        return [*existing, *(event for event in events if event is not None)]

    def _echo(self, event: TelemetryEvent) -> None:
        payload_text = ""
        if event.payload:
            rendered = json.dumps(event.payload, ensure_ascii=True, default=str)
            max_chars = self._ECHO_CHARS_BY_VERBOSITY.get(self._verbosity, 220)
            if len(rendered) > max_chars:
                rendered = f"{rendered[: max_chars - 3]}..."
            payload_text = f" | {rendered}"
        print(
            f"[{event.timestamp}] v{event.verbosity} {event.stage}/{event.payload_type}: {event.message}{payload_text}",
            file=sys.stderr,
            flush=True,
        )

    def _normalize_value(self, value: Any, *, depth: int) -> Any:
        max_depth = self._MAX_DEPTH
        max_items = self._ITEMS_BY_VERBOSITY.get(self._verbosity, 6)
        max_chars = self._CHARS_BY_VERBOSITY.get(self._verbosity, 240)

        if depth >= max_depth:
            return "<trimmed>"
        if hasattr(value, "model_dump"):
            try:
                value = value.model_dump(mode="json")
            except TypeError:
                value = value.model_dump()
        if isinstance(value, Mapping):
            normalized: dict[str, Any] = {}
            items = list(value.items())
            for key, item in items[:max_items]:
                normalized[str(key)] = self._normalize_value(item, depth=depth + 1)
            if len(items) > max_items:
                normalized["_truncated_fields"] = len(items) - max_items
            return normalized
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            normalized_items = [self._normalize_value(item, depth=depth + 1) for item in list(value)[:max_items]]
            if len(value) > max_items:
                normalized_items.append(f"... ({len(value) - max_items} more)")
            return normalized_items
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str):
            compact = " ".join(value.split())
            if len(compact) > max_chars:
                return f"{compact[: max_chars - 3]}..."
            return compact
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if hasattr(value, "value"):
            return value.value
        return str(value)
