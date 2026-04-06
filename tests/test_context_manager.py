from deepresearch.context_manager import ContextManager
from deepresearch.state import (
    ConfidenceLevel,
    CuratedEvidence,
    EvidenceSourceRef,
    Gap,
    GapSeverity,
    ResearchTopic,
    TopicBrief,
    TopicCoverage,
    TopicStatus,
    build_initial_state,
)


def test_synthesis_budget_respects_reporter_reserve(research_config) -> None:
    manager = ContextManager(research_config)
    state = build_initial_state("What is the current state of fusion energy?", max_iterations=4)
    budget = manager.synthesis_budget(state)

    assert budget.context_window_tokens == research_config.model.num_ctx
    assert budget.reserved_output_tokens == min(
        research_config.model.num_predict,
        int(research_config.model.num_ctx * research_config.reporter.output_reserve_ratio),
    )
    assert budget.prompt_margin_tokens == research_config.reporter.prompt_margin_tokens
    assert budget.available_prompt_tokens > 0
    assert budget.final_context_full is False


def test_planner_context_renders_topics_and_gaps(research_config) -> None:
    manager = ContextManager(research_config)
    state = build_initial_state("Explain the 2025 outlook for grid-scale batteries.", max_iterations=4)
    state["plan"] = [
        ResearchTopic(
            id="topic_storage",
            question="How fast is grid storage deployment growing?",
            rationale="Need deployment baseline.",
            evidence_target=1,
            search_terms=["grid storage deployment growth"],
            status=TopicStatus.IN_PROGRESS,
        )
    ]
    state["open_gaps"] = [
        Gap(
            topic_id="topic_storage",
            description="Missing regional deployment comparison",
            severity=GapSeverity.HIGH,
            suggested_queries=["grid battery deployment by region 2024 2025"],
        )
    ]
    state["curated_evidence"] = [
        CuratedEvidence(
            evidence_id="evidence_1",
            topic_id="topic_storage",
            canonical_claim="Battery deployments accelerated in 2024.",
            summary="Deployments accelerated in 2024 across multiple markets.",
            support_quotes=["Deployments accelerated in 2024 across multiple markets."],
            sources=[
                EvidenceSourceRef(
                    url="https://example.com/storage",
                    title="Storage report",
                    locator="p1",
                )
            ],
            exact_generation_tokens=32,
            prompt_fit_tokens_estimate=28,
            confidence=ConfidenceLevel.MEDIUM,
            canonical_fingerprint="battery-deployments-2024",
        )
    ]
    state["topic_coverage"] = {
        "topic_storage": TopicCoverage(
            topic_id="topic_storage",
            accepted_evidence_count=1,
            unique_domains=1,
            attempts=1,
            empty_attempts=0,
            resolved=False,
            exhausted=False,
            rationale="coverage incomplete",
            pending_gaps=["Missing regional deployment comparison"],
        )
    }
    state["working_dossier"].topic_summaries["topic_storage"] = "Battery deployment baseline captured."

    context = manager.planner_context(state)

    assert "topic_storage" in context.active_subqueries
    assert "Missing regional deployment comparison" in context.open_gaps
    assert "topic_storage" in context.coverage_summary
    assert "Battery deployment baseline captured." in context.dossier_context


def test_synthesizer_context_includes_budget_mapping(research_config) -> None:
    manager = ContextManager(research_config)
    state = build_initial_state("Summarize fusion energy commercialization progress.", max_iterations=4)
    state["plan"] = [
        ResearchTopic(
            id="topic_fusion",
            question="What is the commercialization status of fusion energy?",
            rationale="Need main synthesis topic.",
            evidence_target=1,
            search_terms=["fusion commercialization status"],
            status=TopicStatus.COMPLETED,
        )
    ]
    state["curated_evidence"] = [
        CuratedEvidence(
            evidence_id="evidence_fusion",
            topic_id="topic_fusion",
            canonical_claim="Fusion commercialization remains pre-scale.",
            summary="Most efforts remain in pilot or demonstration stages.",
            support_quotes=["Most efforts remain in pilot or demonstration stages."],
            sources=[EvidenceSourceRef(url="https://example.com/fusion", title="Fusion report", locator="p2")],
            exact_generation_tokens=24,
            prompt_fit_tokens_estimate=20,
            confidence=ConfidenceLevel.MEDIUM,
            canonical_fingerprint="fusion-pre-scale",
        )
    ]
    state["synthesis_budget"] = manager.synthesis_budget(state)
    state["topic_briefs"] = {
        "topic_fusion": TopicBrief(
            topic_id="topic_fusion",
            question="What is the commercialization status of fusion energy?",
            markdown_brief=(
                "### Topic\nWhat is the commercialization status of fusion energy?\n\n"
                "### Answer\nMost efforts remain pre-scale.\n\n"
                "### Evidence Highlights\n- Demonstration-stage programs dominate.\n\n"
                "### Uncertainty And Gaps\n- Timelines remain uncertain."
            ),
        )
    }

    context = manager.synthesizer_context(state)

    assert context.query == state["query"]
    assert context.evidentiary[0].evidence_id == "evidence_fusion"
    assert "Most efforts remain pre-scale." in context.topic_briefs_context
    assert state["synthesis_budget"].available_prompt_tokens > 0
    assert state["synthesis_budget"].final_context_full is False
