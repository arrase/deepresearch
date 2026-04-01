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
    graph.add_conditional_edges("source_manager", lambda s: "browser" if s.get("current_candidate") else "evaluator", {"browser": "browser", "evaluator": "evaluator"})
    graph.add_conditional_edges("browser", _route_after_browser, {"extractor": "extractor", "evaluator": "evaluator", "source_manager": "source_manager"})
    graph.add_edge("extractor", "context_manager")
    graph.add_conditional_edges("context_manager", lambda s: _route_after_context_manager(s, runtime), {"source_manager": "source_manager", "evaluator": "evaluator"})
    graph.add_conditional_edges("evaluator", _route_after_evaluator, {"synthesizer": "synthesizer", "source_manager": "source_manager", "planner": "planner"})
    graph.add_edge("synthesizer", END)
    return graph.compile()

def _route_after_browser(state: ResearchState) -> str:
    res = state.get("current_browser_result")
    if not res: return "source_manager"
    if res.status in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}: return "extractor"
    return "evaluator" if state.get("technical_reason") else "source_manager"


def _route_after_context_manager(state: ResearchState, runtime: ResearchRuntime) -> str:
    batch_size = runtime.config.runtime.eval_batch_size
    visited_since = state.get("urls_visited_since_eval", 0)
    has_new_evidence = bool(state.get("latest_evidence"))
    has_queue = bool(state.get("search_queue"))
    if visited_since < batch_size and has_queue and not has_new_evidence:
        return "source_manager"
    return "evaluator"


def _route_after_evaluator(state: ResearchState) -> str:
    if state["is_sufficient"]:
        return "synthesizer"
    if state.get("technical_reason"):
        return "planner"
    return "planner" if not state["active_subqueries"] else "source_manager"
