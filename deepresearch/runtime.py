"""Research runtime dependencies and containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ResearchConfig
    from .context_manager import ContextManager
    from .core.llm import LLMWorkers
    from .telemetry import TelemetryRecorder


@dataclass
class ResearchRuntime:
    config: ResearchConfig
    context_manager: ContextManager
    llm_workers: LLMWorkers
    browser: object  # LightpandaDockerManager
    search_client: object  # DuckDuckGoSearchClient or TavilySearchClient
    telemetry: TelemetryRecorder
