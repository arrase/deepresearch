"""Modular research nodes for LangGraph."""

from __future__ import annotations

from ..runtime import ResearchRuntime
from .context_manager import ContextManagerNode
from .evaluator import EvaluatorNode
from .extractor import ExtractorNode
from .planner import PlannerNode
from .source_manager import SourceManagerNode
from .synthesizer import SynthesizerNode
from .topic_synthesizer import TopicSynthesizerNode

__all__ = [
    "ContextManagerNode",
    "EvaluatorNode",
    "ExtractorNode",
    "PlannerNode",
    "ResearchNodes",
    "SourceManagerNode",
    "SynthesizerNode",
    "TopicSynthesizerNode",
]


class ResearchNodes:
    """Registry of research nodes."""

    def __init__(self, runtime: ResearchRuntime) -> None:
        self.planner = PlannerNode(runtime)
        self.source_manager = SourceManagerNode(runtime)
        self.extractor = ExtractorNode(runtime)
        self.context_manager = ContextManagerNode(runtime)
        self.evaluator = EvaluatorNode(runtime)
        self.topic_synthesizer = TopicSynthesizerNode(runtime)
        self.synthesizer = SynthesizerNode(runtime)
