"""Research StateGraph assembly."""

from __future__ import annotations

from typing import Protocol, cast

from langgraph.graph import END, START, StateGraph

from .nodes import ResearchNodes
from .runtime import ResearchRuntime
from .state import ResearchState


class CompiledResearchGraph(Protocol):
    def invoke(self, input: ResearchState, config: object | None = None) -> ResearchState: ...


def build_graph(runtime: ResearchRuntime) -> CompiledResearchGraph:
    nodes = ResearchNodes(runtime)
    graph = StateGraph(ResearchState)
    graph.add_node("planner", nodes.planner)
    graph.add_node("source_manager", nodes.source_manager)
    graph.add_node("extractor", nodes.extractor)
    graph.add_node("context_manager", nodes.context_manager)
    graph.add_node("evaluator", nodes.evaluator)
    graph.add_node("topic_synthesizer", nodes.topic_synthesizer)
    graph.add_node("synthesizer", nodes.synthesizer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "source_manager")
    graph.add_conditional_edges(
        "source_manager",
        _route_after_source_manager,
        {"extractor": "extractor", "evaluator": "evaluator"},
    )
    graph.add_edge("extractor", "context_manager")
    graph.add_edge("context_manager", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {"topic_synthesizer": "topic_synthesizer", "source_manager": "source_manager", "planner": "planner"},
    )
    graph.add_edge("topic_synthesizer", "synthesizer")
    graph.add_edge("synthesizer", END)
    return cast(CompiledResearchGraph, graph.compile())


def _route_after_source_manager(state: ResearchState) -> str:
    if state["current_batch"]:
        return "extractor"
    return "evaluator"


def _route_after_evaluator(state: ResearchState) -> str:
    if state.get("stop_reason"):
        return "topic_synthesizer"
    if state.get("replan_requested") or not state["plan"]:
        return "planner"
    return "source_manager"
