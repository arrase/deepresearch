from deepresearch.context_manager import ContextManager
from deepresearch.state import (
    ConfidenceLevel,
    CuratedEvidence,
    EvidenceSourceRef,
    Gap,
    GapSeverity,
    ResearchTopic,
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


def test_meta_planner_context_renders_hypotheses(research_config) -> None:
    manager = ContextManager(research_config)
    state = build_initial_state("Explain the 2025 outlook for grid-scale batteries.", max_iterations=4)
    state["hypotheses"] = ["Batteries are getting cheaper", "Deployment is growing"]

    context = manager.meta_planner_context(state)

    assert "Batteries are getting cheaper" in context.hypotheses
    assert context.max_chapters == research_config.runtime.max_chapters
    assert context.min_chapters >= 1


def test_micro_planner_context_renders_chapter_topics_and_gaps(research_config) -> None:
    manager = ContextManager(research_config)
    state = build_initial_state("Explain the 2025 outlook for grid-scale batteries.", max_iterations=4)
    chapter = ResearchTopic(
        id="ch_storage",
        question="How fast is grid storage deployment growing?",
        rationale="Need deployment baseline.",
        evidence_target=1,
        search_terms=["grid storage deployment growth"],
        status=TopicStatus.IN_PROGRESS,
        depth=0,
        chapter_id="ch_storage",
    )
    sub = ResearchTopic(
        id="sub_regional",
        question="Regional comparison of storage deployment",
        rationale="Need regions data",
        evidence_target=1,
        search_terms=["storage region"],
        status=TopicStatus.PENDING,
        depth=1,
        parent_id="ch_storage",
        chapter_id="ch_storage",
    )
    state["plan"] = [chapter, sub]
    state["current_chapter_id"] = "ch_storage"
    state["open_gaps"] = [
        Gap(
            topic_id="sub_regional",
            description="Missing regional deployment comparison",
            severity=GapSeverity.HIGH,
            suggested_queries=["grid battery deployment by region 2024 2025"],
        )
    ]
    state["curated_evidence"] = [
        CuratedEvidence(
            evidence_id="evidence_1",
            topic_id="sub_regional",
            chapter_id="ch_storage",
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
        "ch_storage": TopicCoverage(
            topic_id="ch_storage",
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
    state["working_dossier"].topic_summaries["sub_regional"] = "Battery deployment baseline captured."

    context = manager.micro_planner_context(state, chapter)

    assert context.chapter_id == "ch_storage"
    assert context.chapter_question == chapter.question
    assert "Missing regional deployment comparison" in context.open_gaps


def test_global_synthesizer_context_includes_chapter_drafts(research_config) -> None:
    from deepresearch.state import ChapterDraft

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
            depth=0,
            chapter_id="topic_fusion",
        )
    ]
    state["curated_evidence"] = [
        CuratedEvidence(
            evidence_id="evidence_fusion",
            topic_id="topic_fusion",
            chapter_id="topic_fusion",
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
    state["chapter_drafts"] = [
        ChapterDraft(
            chapter_id="topic_fusion",
            title="Fusion Commercialization",
            executive_summary="Most efforts remain in pilot stages.",
            key_findings=["Pre-scale"],
            evidence_ids=["evidence_fusion"],
        )
    ]
    state["synthesis_budget"] = manager.synthesis_budget(state)

    context = manager.global_synthesizer_context(state)

    assert context.query == state["query"]
    assert len(context.chapters) == 1
    assert context.chapters[0]["title"] == "Fusion Commercialization"
    assert state["synthesis_budget"].available_prompt_tokens > 0
    assert state["synthesis_budget"].final_context_full is False
