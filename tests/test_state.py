from deepresearch.state import build_initial_state


def test_build_initial_state_has_required_fields() -> None:
    state = build_initial_state(
        "What is the market outlook for fusion energy?",
        max_iterations=6,
    )
    assert state["query"].startswith("What is")
    assert state["max_iterations"] == 6
    assert state["atomic_evidence"] == []
    assert state["llm_usage"] == {}
    assert state["synthesis_budget"] == {}
    assert state["stop_reason"] is None
    assert state["technical_reason"] is None
    assert state["stagnation_cycles"] == 0
    assert state["consecutive_technical_failures"] == 0
    assert state["cycles_without_new_evidence"] == 0
    assert state["cycles_without_useful_sources"] == 0
    assert state["progress_score"] == 0
    assert state["useful_sources_count"] == 0
