"""Base utilities and decorators for research nodes."""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from ..state import ResearchState

F = TypeVar("F", bound=Callable[..., Any])


def consume_llm_telemetry_events(runtime: Any) -> list[Any]:
    consumer = getattr(runtime.llm_workers, "consume_telemetry_events", None)
    if callable(consumer):
        return consumer()
    return []


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
                verbosity=1,
                payload_type="lifecycle",
                **kwargs.get("telemetry_payload", {}),
            )
            
            # Update state with telemetry
            state_with_telemetry = {
                **state,
                "telemetry": self._runtime.telemetry.extend(state.get("telemetry", []), start_event),
            }
            
            try:
                # Execute the node logic
                result = func(self, state_with_telemetry, *args, **kwargs)
                return result
            except Exception as e:
                llm_events = consume_llm_telemetry_events(self._runtime)
                # Record error event
                error_event = self._runtime.telemetry.record(
                    stage,
                    f"Error in {stage}: {str(e)}",
                    verbosity=1,
                    payload_type="error",
                    error=str(e),
                )
                return {
                    **state_with_telemetry,
                    "telemetry": self._runtime.telemetry.extend(state_with_telemetry["telemetry"], *llm_events, error_event),
                }
        return wrapper  # type: ignore
    return decorator
