from deepresearch.context_manager import ContextManager
from deepresearch.core.utils import canonicalize_url
from deepresearch.graph import build_graph
from deepresearch.nodes import SourceManagerNode
from deepresearch.runtime import ResearchRuntime
from deepresearch.state import (
    ResearchState,
    ResearchTopic,
    SearchCandidate,
    StopReason,
    TopicCoverage,
    TopicStatus,
    build_initial_state,
)
from tests.conftest import (
    EmptySearchClient,
    ExhaustedLLMWorkers,
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
    # The chapter (depth==0) topic should be in the plan
    chapters = [t for t in result["plan"] if t.depth == 0]
    assert len(chapters) >= 1


def test_graph_stops_when_search_finds_nothing(research_config) -> None:
    research_config = research_config.model_copy(deep=True)
    research_config.runtime.min_attempts_before_exhaustion = 1
    runtime = ResearchRuntime(
        config=research_config,
        llm_workers=ExhaustedLLMWorkers(),
        search_client=EmptySearchClient(),
        context_manager=ContextManager(research_config),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "Assess the state of industrial heat pumps.",
        max_iterations=4,
    )

    result = graph.invoke(initial_state)

    assert result["stop_reason"] in {
        StopReason.PLAN_COMPLETED,
        StopReason.STUCK_NO_SOURCES,
        StopReason.MAX_ITERATIONS_REACHED,
    }
    assert result["final_report"] is not None


def test_graph_marks_context_saturation_when_budget_full(research_config) -> None:
    research_config = research_config.model_copy(deep=True)
    research_config.model.num_ctx = 1500
    research_config.model.num_predict = 1500
    research_config.reporter.output_reserve_ratio = 0.7
    research_config.reporter.prompt_margin_tokens = 100

    runtime = ResearchRuntime(
        config=research_config,
        llm_workers=FinalContextFullLLMWorkers(),
        search_client=FakeSearchClient(),
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


def test_graph_blocked_source_does_not_create_evidence_or_sources(research_config) -> None:
    class BlockedSearchClient:
        def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
            return [
                SearchCandidate(
                    url="https://www.linkedin.com/posts/example",
                    normalized_url="https://www.linkedin.com/posts/example",
                    title="LinkedIn activity",
                    snippet="Robots blocked",
                    domain="linkedin.com",
                    raw_content='$time=1775261612956 $scope=http $level=warn $msg="blocked by robots"',
                )
            ]

    runtime = ResearchRuntime(
        config=research_config,
        llm_workers=ExhaustedLLMWorkers(),
        search_client=BlockedSearchClient(),
        context_manager=ContextManager(research_config),
    )
    graph = build_graph(runtime)
    initial_state = build_initial_state(
        "Compare browser automation stacks",
        max_iterations=1,
    )

    result = graph.invoke(initial_state)

    assert result["stop_reason"] == StopReason.MAX_ITERATIONS_REACHED
    assert result["curated_evidence"] == []
    assert result["final_report"] is not None
    assert result["final_report"].cited_sources == []
    assert "blocked by robots" not in result["final_report"].markdown_report.lower()


def test_source_manager_prefers_article_over_feed_for_news_query(research_config) -> None:
    class MixedNewsSearchClient:
        def search(self, query: str, *, max_results: int | None = None) -> list:
            return [
                SearchCandidate(
                    url="https://gcdiario.com/seccion/sucesos/feed",
                    normalized_url="https://gcdiario.com/seccion/sucesos/feed",
                    title="SUCESOS archivos - GC Diario",
                    snippet="Sucesos en castellano y Galicia",
                    domain="gcdiario.com",
                    raw_content=(
                        "Sucesos de ultima hora en Galicia y otras regiones. "
                        "Resumen de titulares agregados sin detalle del incidente. "
                        "Sucesos de ultima hora en Galicia y otras regiones. "
                        "Resumen de titulares agregados sin detalle del incidente. "
                    ),
                ),
                SearchCandidate(
                    url="https://castellonplaza.com/sucesos/castellon-detencion-ayer-centro",
                    normalized_url="https://castellonplaza.com/sucesos/castellon-detencion-ayer-centro",
                    title="Detenido un hombre tras un altercado en Castellon",
                    snippet="Sucesos de ayer en Castellon con intervencion policial en el centro.",
                    domain="castellonplaza.com",
                    raw_content=(
                        "La noticia detalla un altercado ocurrido ayer en el centro de Castellon "
                        "con intervencion policial, "
                        "testigos, contexto del incidente y consecuencias posteriores para las personas implicadas. "
                        "La noticia detalla un altercado ocurrido ayer en el centro de Castellon "
                        "con intervencion policial, "
                        "testigos, contexto del incidente y consecuencias posteriores para las personas implicadas. "
                    ),
                ),
            ]

    runtime = ResearchRuntime(
        config=research_config,
        llm_workers=ExhaustedLLMWorkers(),
        search_client=MixedNewsSearchClient(),
        context_manager=ContextManager(research_config),
    )
    node = SourceManagerNode(runtime)
    initial_state: ResearchState = build_initial_state(
        "resume las ultimas noticias de sucesos de ayer en castellon",
        max_iterations=4,
    )
    topic = ResearchTopic(
        id="topic_news",
        question="resume las ultimas noticias de sucesos de ayer en castellon",
        rationale="Need local incidents coverage",
        search_terms=["sucesos castellon ayer"],
        status=TopicStatus.PENDING,
        depth=0,
        chapter_id="topic_news",
    )
    initial_state["plan"] = [topic]
    initial_state["current_chapter_id"] = "topic_news"
    initial_state["topic_coverage"] = {"topic_news": TopicCoverage(topic_id="topic_news")}

    result = node(initial_state)

    assert result["current_batch"]
    assert result["current_batch"][0].url == "https://castellonplaza.com/sucesos/castellon-detencion-ayer-centro"
    assert any(item.url == "https://gcdiario.com/seccion/sucesos/feed" for item in result["discarded_sources"])
    expected_url = canonicalize_url("https://castellonplaza.com/sucesos/castellon-detencion-ayer-centro")
    assert expected_url in result["visited_urls"]
