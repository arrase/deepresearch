"""Ensamblaje del StateGraph de investigacion."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from nodes import ResearchNodes, ResearchRuntime
from state import BrowserPageStatus, ResearchState


def build_graph(runtime: ResearchRuntime):
    nodes = ResearchNodes(runtime)
    graph = StateGraph(ResearchState)
    graph.add_node("planner", nodes.node_planner)
    graph.add_node("source_manager", nodes.node_source_manager)
    graph.add_node("browser", nodes.node_browser)
    graph.add_node("extractor", nodes.node_extractor)
    graph.add_node("context_manager", nodes.node_context_manager)
    graph.add_node("evaluator", nodes.node_evaluator)
    graph.add_node("synthesizer", nodes.node_synthesizer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "source_manager")
    graph.add_edge("source_manager", "browser")
    graph.add_conditional_edges(
        "browser",
        _route_after_browser,
        {
            "extractor": "extractor",
            "evaluator": "evaluator",
            "source_manager": "source_manager",
        },
    )
    graph.add_edge("extractor", "context_manager")
    graph.add_edge("context_manager", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {
            "synthesizer": "synthesizer",
            "source_manager": "source_manager",
        },
    )
    graph.add_edge("synthesizer", END)
    return graph.compile()


def _route_after_browser(state: ResearchState) -> str:
    browser_result = state.get("current_browser_result")
    if browser_result is None:
        return "source_manager"
    if state["fallback_reason"] and not state.get("current_candidate"):
        return "evaluator"
    if browser_result.status in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
        return "extractor"
    if state["fallback_reason"] and state["iteration"] >= state["max_iterations"]:
        return "evaluator"
    return "source_manager"


def _route_after_evaluator(state: ResearchState) -> str:
    if state["is_sufficient"]:
        return "synthesizer"
    return "source_manager"
