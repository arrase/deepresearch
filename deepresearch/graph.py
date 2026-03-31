"""Research StateGraph assembly."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .runtime import ResearchRuntime
from .nodes import ResearchNodes
from .state import BrowserPageStatus, ResearchState


def build_graph(runtime: ResearchRuntime):
    nodes = ResearchNodes(runtime)
    graph = StateGraph(ResearchState)
    graph.add_node("planner", nodes.planner)
    graph.add_node("source_manager", nodes.source_manager)
    graph.add_node("browser", nodes.browser)
    graph.add_node("extractor", nodes.extractor)
    graph.add_node("context_manager", nodes.context_manager)
    graph.add_node("evaluator", nodes.evaluator)
    graph.add_node("synthesizer", nodes.synthesizer)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "source_manager")
    graph.add_conditional_edges(
        "source_manager",
        _route_after_source_manager,
        {
            "browser": "browser",
            "evaluator": "evaluator",
        },
    )
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
            "planner": "planner",
        },
    )
    graph.add_edge("synthesizer", END)
    return graph.compile()


def _route_after_source_manager(state: ResearchState) -> str:
    if state.get("current_candidate") is None:
        return "evaluator"
    return "browser"


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
    
    if not state["active_subqueries"]:
        return "planner"
        
    return "source_manager"
