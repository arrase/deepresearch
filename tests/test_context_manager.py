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
    context = manager.extractor_context(state, target_subquery_ids=[subquery.id], local_source="Content")
    assert context.evidentiary
    assert context.local_source == "Content"
