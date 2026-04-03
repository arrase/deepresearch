"""Logging and LangSmith tracing helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from langsmith.run_helpers import tracing_context

from .config import ResearchConfig

_LOGGER_NAME = "deepresearch"
_VERBOSITY_TO_LEVEL = {
    0: logging.WARNING,
    1: logging.INFO,
    2: logging.DEBUG,
    3: logging.DEBUG,
}


def configure_logging(verbosity: int) -> None:
    """Configure application logging from runtime verbosity."""
    logging.basicConfig(
        level=_VERBOSITY_TO_LEVEL.get(verbosity, logging.DEBUG),
        format="%(message)s",
        force=True,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    if name is None:
        return logging.getLogger(_LOGGER_NAME)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


def should_log(config: ResearchConfig, verbosity: int = 1) -> bool:
    return config.runtime.verbosity >= verbosity


@contextmanager
def langsmith_tracing(config: ResearchConfig, *, metadata: dict[str, Any] | None = None):
    """Enable LangSmith tracing for the current execution when configured."""
    langsmith = config.langsmith
    if not (langsmith.enabled and langsmith.tracing):
        yield
        return

    client_kwargs: dict[str, Any] = {"api_key": langsmith.api_key}
    if langsmith.endpoint:
        client_kwargs["api_url"] = langsmith.endpoint
    client = Client(**client_kwargs)
    with tracing_context(
        enabled=True,
        client=client,
        project_name=langsmith.project,
        tags=["deepresearch"],
        metadata=metadata,
    ):
        try:
            yield
        finally:
            wait_for_all_tracers()
