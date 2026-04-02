"""Research runtime dependencies and containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import ResearchConfig
    from .context_manager import ContextManager
    from .core.llm import LLMWorkers
    from .telemetry import TelemetryRecorder
    from .tools import DuckDuckGoSearchClient, LightpandaDockerManager, TavilySearchClient


@dataclass
class ResearchRuntime:
    config: ResearchConfig
    context_manager: ContextManager
    llm_workers: LLMWorkers
    browser: LightpandaDockerManager
    search_client: DuckDuckGoSearchClient | TavilySearchClient
    telemetry: TelemetryRecorder
