from deepresearch.state import SynthesisBudget, build_initial_state


def test_build_initial_state_has_required_fields() -> None:
    state = build_initial_state(
        "What is the market outlook for fusion energy?",
        max_iterations=6,
    )
    assert state["query"].startswith("What is")
    assert state["max_iterations"] == 6
    assert state["plan"] == []
    assert state["active_topic_id"] is None
    assert state["curated_evidence"] == []
    assert state["llm_usage"] == {}
    assert isinstance(state["synthesis_budget"], SynthesisBudget)
    assert state["synthesis_budget"].final_context_full is False
    assert state["topic_briefs"] == {}
    assert state["stop_reason"] is None
    assert state["technical_reason"] is None
    assert state["cycles_without_new_evidence"] == 0
    assert state["cycles_without_useful_sources"] == 0
    assert state["consecutive_technical_failures"] == 0
    assert state["new_evidence_in_cycle"] == 0
    assert state["useful_source_in_cycle"] is False
