"""Base logging helpers for research nodes."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar, cast

from ..observability import get_logger
from ..state import ResearchState

F = TypeVar("F", bound=Callable[..., Any])
logger = get_logger("pipeline")


def log_runtime_event(runtime: Any, message: str, *, verbosity: int = 1, **payload: Any) -> None:
    if runtime.config.runtime.verbosity < verbosity:
        return

    rendered_payload = ""
    if payload:
        rendered_payload = json.dumps(payload, ensure_ascii=True, default=str)
        if len(rendered_payload) > 1000:
            rendered_payload = f"{rendered_payload[:997]}..."

    level = logging.INFO if verbosity == 1 else logging.DEBUG
    if rendered_payload:
        logger.log(level, "%s | %s", message, rendered_payload)
    else:
        logger.log(level, message)


def log_node_activity(stage: str, message_template: str) -> Callable[[F], F]:
    """Decorator to emit concise start/error logs for a node."""

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(self, state: ResearchState, *args: Any, **kwargs: Any) -> dict:
            try:
                message = message_template.format(query=state.get("query", "unknown"))
            except (KeyError, IndexError):
                message = message_template

            log_runtime_event(self._runtime, f"[{stage}] {message}", verbosity=1)
            try:
                return cast(dict[str, object], func(self, state, *args, **kwargs))
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                log_runtime_event(self._runtime, f"[{stage}] Error", verbosity=1, error=str(exc))
                raise

        return wrapper  # type: ignore[return-value]

    return decorator
