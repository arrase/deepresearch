from deepresearch.config import ResearchConfig
from deepresearch.context_manager import ContextManager
from deepresearch.state import AtomicEvidence, Subquery, build_initial_state


def test_extractor_context_selects_relevant_evidence() -> None:
    config = ResearchConfig()
    manager = ContextManager(config)
    state = build_initial_state(
        "Primary question",
        max_iterations=4,
    )
    subquery = Subquery(question="Subquery", rationale="r", search_terms=["fusion"])
    state["active_subqueries"] = [subquery]
    state["atomic_evidence"] = [
        AtomicEvidence(
            subquery_id=subquery.id,
            source_url="https://example.com",
            source_title="Example",
            summary="Resumen",
            claim="Fusion plants remain expensive",
            quotation="Fusion plants remain expensive",
            citation_locator="p2",
        )
    ]
    context = manager.extractor_context(state, targets=[subquery.id], local_source="Content")
    assert context.evidentiary
    assert context.local_source == "Content"


def test_synthesis_budget_marks_final_context_as_full_when_evidence_overflows() -> None:
    config = ResearchConfig()
    config.model.num_ctx = 200
    config.model.num_predict = 64
    config.runtime.synthesizer_output_reserve_ratio = 0.20
    config.runtime.synthesizer_prompt_margin = 0
    manager = ContextManager(config)
    state = build_initial_state("Primary question", max_iterations=4)
    subquery = Subquery(question="Subquery", rationale="r", search_terms=["fusion"])
    state["active_subqueries"] = [subquery]
    state["atomic_evidence"] = [
        AtomicEvidence(
            subquery_id=subquery.id,
            source_url=f"https://example.com/{idx}",
            source_title=f"Example {idx}",
            summary="Resumen " * 8,
            claim=f"Fusion plants remain expensive {idx}",
            quotation=("Fusion plants remain expensive and complex. " * 10).strip(),
            citation_locator="p2",
        )
        for idx in range(3)
    ]

    budget = manager.synthesis_budget(state)

    assert budget["final_context_full"] is True
    candidate_count = budget["candidate_evidence_count"]
    selected_count = budget["selected_evidence_count"]
    assert isinstance(candidate_count, int)
    assert isinstance(selected_count, int)
    assert candidate_count >= selected_count


def test_planner_context_surfaces_coverage_and_source_balance_signals() -> None:
    config = ResearchConfig()
    manager = ContextManager(config)
    state = build_initial_state("How should I evaluate a new browser automation tool?", max_iterations=4)
    sq_1 = Subquery(
        id="sq_1",
        question="What capabilities does it provide?",
        rationale="r",
        search_terms=["capabilities"],
        evidence_target=2,
    )
    sq_2 = Subquery(
        id="sq_2",
        question="What are the limitations and trade-offs?",
        rationale="r",
        search_terms=["limitations"],
        evidence_target=2,
    )
    state["active_subqueries"] = [sq_1, sq_2]
    state["atomic_evidence"] = [
        AtomicEvidence(
            subquery_id="sq_1",
            source_url="https://vendor.example/docs",
            source_title="Vendor docs",
            summary="Capabilities summary",
            claim="The tool supports browser automation through a scripting API.",
            quotation="The tool supports browser automation through a scripting API.",
            citation_locator="p1",
        ),
        AtomicEvidence(
            subquery_id="sq_1",
            source_url="https://vendor.example/blog",
            source_title="Vendor blog",
            summary="Capabilities summary",
            claim="The tool emphasizes speed and lightweight execution.",
            quotation="The tool emphasizes speed and lightweight execution.",
            citation_locator="p2",
        ),
    ]

    context = manager.planner_context(state)

    assert "sq_2" in context.coverage_summary
    assert "no evidence yet" in context.coverage_summary
    assert "vendor.example" in context.source_balance_summary
    assert "highly concentrated" in context.source_balance_summary
