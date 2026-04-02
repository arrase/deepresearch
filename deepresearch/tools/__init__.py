"""External integrations: search backends and browser adapters.

This package provides the search and browser implementations used by the
research pipeline.  Re-exports preserve backward compatibility for any
``from deepresearch.tools import …`` imports.
"""

from __future__ import annotations

from .duckduckgo import DuckDuckGoSearchClient
from .lightpanda import LightpandaDockerManager
from .tavily import TavilySearchClient

__all__ = [
    "DuckDuckGoSearchClient",
    "LightpandaDockerManager",
    "TavilySearchClient",
]
