from deepresearch.state import build_initial_state


def test_build_initial_state_has_required_fields() -> None:
    state = build_initial_state(
        "What is the market outlook for fusion energy?",
        max_iterations=6,
        target_tokens=100000,
        configured_by="test",
        selection_policy="hierarchical_relevance_first",
    )
    assert state["query"].startswith("What is")
    assert state["max_iterations"] == 6
    assert state["context_window_config"].target_tokens == 100000
    assert state["atomic_evidence"] == []
