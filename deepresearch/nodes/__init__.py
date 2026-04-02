"""Modular research nodes for LangGraph."""

from __future__ import annotations

from ..runtime import ResearchRuntime
from .browser import BrowserNode
from .context_manager import ContextManagerNode
from .evaluator import EvaluatorNode
from .extractor import ExtractorNode
from .planner import PlannerNode
from .source_manager import SourceManagerNode
from .synthesizer import SynthesizerNode

__all__ = [
    "BrowserNode",
    "ContextManagerNode",
    "EvaluatorNode",
    "ExtractorNode",
    "PlannerNode",
    "ResearchNodes",
    "SourceManagerNode",
    "SynthesizerNode",
]


class ResearchNodes:
    """Registry of research nodes."""

    def __init__(self, runtime: ResearchRuntime) -> None:
        self.planner = PlannerNode(runtime)
        self.source_manager = SourceManagerNode(runtime)
        self.browser = BrowserNode(runtime)
        self.extractor = ExtractorNode(runtime)
        self.context_manager = ContextManagerNode(runtime)
        self.evaluator = EvaluatorNode(runtime)
        self.synthesizer = SynthesizerNode(runtime)
