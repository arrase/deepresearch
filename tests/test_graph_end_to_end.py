from deepresearch.context_manager import ContextManager
from deepresearch.core.payloads import PlannerPayload
from deepresearch.graph import build_graph
from deepresearch.runtime import ResearchRuntime
from deepresearch.state import ResearchTopic, SearchIntent, StopReason, TopicStatus, build_initial_state
from tests.conftest import (
    EmptySearchClient,
    ExhaustedLLMWorkers,
    FakeBrowser,
    FakeSearchClient,
    FinalContextFullLLMWorkers,
)


def test_graph_reaches_plan_completed(fake_runtime) -> None:
    graph = build_graph(fake_runtime)
    initial_state = build_initial_state(
        "Assess the state of industrial heat pumps.",
        max_iterations=6,
    )

    result = graph.invoke(initial_state)

    assert result["final_report"] is not None
    assert result["stop_reason"] == StopReason.PLAN_COMPLETED
    assert result["curated_evidence"]
    assert result["topic_coverage"]["topic_demo"].accepted_evidence_count >= 1
    assert result["plan"][0].status == TopicStatus.COMPLETED


def test_graph_stops_when_search_finds_nothing(research_config) -> None:
    runtime = ResearchRuntime(
        config=research_config,
        llm_workers=ExhaustedLLMWorkers(),
        search_client=EmptySearchClient(),
        browser=FakeBrowser(),
        context_manager=ContextManager(research_config),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "Assess the state of industrial heat pumps.",
        max_iterations=4,
    )

    result = graph.invoke(initial_state)

    assert result["stop_reason"] == StopReason.PLAN_COMPLETED
    assert result["final_report"] is not None
    assert result["curated_evidence"] == []
    assert result["plan"][0].status == TopicStatus.EXHAUSTED


def test_graph_marks_context_saturation_when_budget_full(research_config) -> None:
    class ContextSaturationWorkers(FinalContextFullLLMWorkers):
        def plan_research(self, context: object) -> PlannerPayload:
            topic = ResearchTopic(
                id="topic_demo",
                question="What happened to industrial heat pump deployment?",
                rationale="Need primary claim",
                evidence_target=1,
                search_terms=["industrial heat pump deployment"],
                success_criteria=[
                    "Need at least one accepted evidence item and unresolved gap review"
                ],
                status=TopicStatus.PENDING,
            )
            return PlannerPayload(
                subqueries=[topic],
                search_intents=[
                    SearchIntent(
                        query="industrial heat pump deployment",
                        rationale="primary",
                        topic_ids=[topic.id],
                    )
                ],
                hypotheses=["Deployment is increasing"],
            )

        def plan_research_with_usage(self, context: object) -> tuple[PlannerPayload, dict[str, int]]:
            return self.plan_research(context), {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }

    research_config = research_config.model_copy(deep=True)
    research_config.model.num_ctx = 1500
    research_config.model.num_predict = 1500
    research_config.reporter.output_reserve_ratio = 0.7
    research_config.reporter.prompt_margin_tokens = 100

    runtime = ResearchRuntime(
        config=research_config,
        llm_workers=ContextSaturationWorkers(),
        search_client=FakeSearchClient(),
        browser=FakeBrowser(),
        context_manager=ContextManager(research_config),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "Assess the state of industrial heat pumps.",
        max_iterations=6,
    )

    result = graph.invoke(initial_state)

    assert result["stop_reason"] == StopReason.CONTEXT_SATURATION
    assert result["final_report"] is not None
    assert result["synthesis_budget"].final_context_full is True
