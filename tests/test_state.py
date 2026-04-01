from deepresearch.state import TelemetryEvent, build_initial_state


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


def test_telemetry_event_supports_structured_verbosity_metadata() -> None:
    event = TelemetryEvent(
        stage="planner",
        message="Agenda updated",
        verbosity=2,
        payload_type="llm_decision",
        payload={"new": 2},
    )

    assert event.verbosity == 2
    assert event.payload_type == "llm_decision"
    assert event.payload["new"] == 2
