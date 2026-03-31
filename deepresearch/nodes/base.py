"""Base utilities and decorators for research nodes."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from ..state import ResearchState

F = TypeVar("F", bound=Callable[..., Any])


def record_telemetry(stage: str, message_template: str) -> Callable[[F], F]:
    """Decorator to record telemetry events for a node.
    
    It records a 'start' event and handles potential errors by recording them
    in the state.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, state: ResearchState, *args: Any, **kwargs: Any) -> dict:
            # Format message if template contains placeholders
            try:
                message = message_template.format(query=state.get("query", "unknown"))
            except (KeyError, IndexError):
                message = message_template

            # Record start event
            start_event = self._runtime.telemetry.record(
                stage,
                message,
                **kwargs.get("telemetry_payload", {}),
            )
            
            # Update state with telemetry
            state_with_telemetry = {**state, "telemetry": [*state.get("telemetry", []), start_event]}
            
            try:
                # Execute the node logic
                result = func(self, state_with_telemetry, *args, **kwargs)
                return result
            except Exception as e:
                # Record error event
                error_event = self._runtime.telemetry.record(
                    stage,
                    f"Error in {stage}: {str(e)}",
                    error=True,
                )
                return {
                    **state_with_telemetry,
                    "telemetry": [*state_with_telemetry["telemetry"], error_event],
                }
        return wrapper  # type: ignore
    return decorator
