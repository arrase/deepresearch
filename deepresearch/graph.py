"""Research StateGraph assembly — hierarchical Map-Reduce architecture."""

from __future__ import annotations

from typing import Protocol, cast

from langgraph.graph import END, START, StateGraph

from .nodes import ResearchNodes
from .runtime import ResearchRuntime
from .state import ResearchState, TopicStatus


class CompiledResearchGraph(Protocol):
    def invoke(self, input: ResearchState, config: object | None = None) -> ResearchState: ...


def build_graph(runtime: ResearchRuntime) -> CompiledResearchGraph:
    nodes = ResearchNodes(runtime)
    graph = StateGraph(ResearchState)

    graph.add_node("meta_planner", nodes.meta_planner)
    graph.add_node("micro_planner", nodes.micro_planner)
    graph.add_node("source_manager", nodes.source_manager)
    graph.add_node("extractor", nodes.extractor)
    graph.add_node("context_manager", nodes.context_manager)
    graph.add_node("evaluator", nodes.evaluator)
    graph.add_node("auditor", nodes.auditor)
    graph.add_node("sub_synthesizer", nodes.sub_synthesizer)
    graph.add_node("global_synthesizer", nodes.global_synthesizer)

    # Flow: START → meta_planner → micro_planner → source_manager → ...
    graph.add_edge(START, "meta_planner")
    graph.add_edge("meta_planner", "micro_planner")
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
        {
            "source_manager": "source_manager",
            "auditor": "auditor",
            "sub_synthesizer": "sub_synthesizer",
        },
    )
    graph.add_edge("micro_planner", "source_manager")
    graph.add_conditional_edges(
        "auditor",
        _route_after_auditor,
        {"sub_synthesizer": "sub_synthesizer", "micro_planner": "micro_planner"},
    )
    graph.add_conditional_edges(
        "sub_synthesizer",
        _route_after_sub_synthesizer,
        {"global_synthesizer": "global_synthesizer", "micro_planner": "micro_planner"},
    )
    graph.add_edge("global_synthesizer", END)

    return cast(CompiledResearchGraph, graph.compile())


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def _route_after_source_manager(state: ResearchState) -> str:
    if state["current_batch"]:
        return "extractor"
    return "evaluator"


def _route_after_evaluator(state: ResearchState) -> str:
    """Route after the deterministic evaluator.

    - Global stop (max_iterations, stuck) → force sub_synthesizer for current chapter.
    - All chapter topics resolved → auditor.
    - Otherwise → source_manager (keep researching).
    """
    if state.get("stop_reason"):
        return "sub_synthesizer"

    chapter_id = state.get("current_chapter_id") or ""
    if chapter_id:
        chapter_topics = [t for t in state["plan"] if t.chapter_id == chapter_id]
        # Only check sub-topics when they exist; depth=0 is a container
        subs = [t for t in chapter_topics if t.depth > 0]
        if subs:
            chapter_topics = subs
        all_done = bool(chapter_topics) and all(
            t.status in {TopicStatus.COMPLETED, TopicStatus.EXHAUSTED} for t in chapter_topics
        )
        if all_done:
            return "auditor"

    return "source_manager"


def _route_after_auditor(state: ResearchState) -> str:
    if state.get("audit_approved", False):
        return "sub_synthesizer"
    return "micro_planner"


def _route_after_sub_synthesizer(state: ResearchState) -> str:
    """After chapter synthesis, either move to the next chapter or finish."""
    chapters = [t for t in state["plan"] if t.depth == 0]
    completed = set(state.get("completed_chapter_ids") or [])
    all_done = bool(chapters) and all(ch.id in completed for ch in chapters)
    if all_done or state.get("stop_reason"):
        return "global_synthesizer"
    return "micro_planner"
