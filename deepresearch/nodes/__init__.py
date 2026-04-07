"""Modular research nodes for LangGraph."""

from __future__ import annotations

from ..runtime import ResearchRuntime
from .auditor import AuditorNode
from .context_manager import ContextManagerNode
from .evaluator import EvaluatorNode
from .extractor import ExtractorNode
from .global_synthesizer import GlobalSynthesizerNode
from .meta_planner import MetaPlannerNode
from .micro_planner import MicroPlannerNode
from .source_manager import SourceManagerNode
from .sub_synthesizer import SubSynthesizerNode

__all__ = [
    "AuditorNode",
    "ContextManagerNode",
    "EvaluatorNode",
    "ExtractorNode",
    "GlobalSynthesizerNode",
    "MetaPlannerNode",
    "MicroPlannerNode",
    "ResearchNodes",
    "SourceManagerNode",
    "SubSynthesizerNode",
]


class ResearchNodes:
    """Registry of research nodes."""

    def __init__(self, runtime: ResearchRuntime) -> None:
        self.meta_planner = MetaPlannerNode(runtime)
        self.micro_planner = MicroPlannerNode(runtime)
        self.source_manager = SourceManagerNode(runtime)
        self.extractor = ExtractorNode(runtime)
        self.context_manager = ContextManagerNode(runtime)
        self.evaluator = EvaluatorNode(runtime)
        self.auditor = AuditorNode(runtime)
        self.sub_synthesizer = SubSynthesizerNode(runtime)
        self.global_synthesizer = GlobalSynthesizerNode(runtime)
