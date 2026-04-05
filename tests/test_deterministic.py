from deepresearch.core.utils import (
    canonicalize_url,
    choose_active_topic,
    compute_minimum_coverage,
    deduplicate_candidates,
    rank_topics_for_source,
    validate_candidate_for_topic,
)
from deepresearch.state import (
    CuratedEvidence,
    EvidenceSourceRef,
    ResearchTopic,
    SearchCandidate,
    TopicCoverage,
    TopicStatus,
)


def test_canonicalize_url_removes_tracking_query_and_fragment() -> None:
    url = "https://Example.com/path/?utm_source=x&id=42#fragment"
    assert canonicalize_url(url) == "https://example.com/path?id=42"


def test_compute_minimum_coverage_marks_topic_resolved() -> None:
    topic = ResearchTopic(question="What is X?", rationale="Need definition", evidence_target=1)
    evidence = [
        CuratedEvidence(
            topic_id=topic.id,
            canonical_claim="X is a defined concept",
            summary="X is defined",
            support_quotes=["X means ..."],
            sources=[EvidenceSourceRef(url="https://example.com", title="Example", locator="p1")],
        )
    ]

    resolved, gaps = compute_minimum_coverage([topic], evidence)

    assert resolved == [topic.id]
    assert gaps == []


def test_deduplicate_candidates_merges_topic_ids() -> None:
    candidates, discarded, repeated = deduplicate_candidates(
        [
            SearchCandidate(url="https://example.com/path", title="Example", topic_ids=["topic_1"], query="q1"),
            SearchCandidate(
                url="https://example.com/path?utm_source=test",
                title="Example",
                topic_ids=["topic_2"],
                query="q2",
            ),
        ],
        visited_urls={},
    )

    assert len(candidates) == 1
    assert candidates[0].topic_ids == ["topic_1", "topic_2"]
    assert candidates[0].query == "q1"
    assert discarded == []
    assert repeated == 1


def test_rank_topics_for_source_prefers_matching_topic() -> None:
    topic_1 = ResearchTopic(
        id="topic_1",
        question="What is Tavily?",
        rationale="r",
        search_terms=["tavily research api"],
    )
    topic_2 = ResearchTopic(
        id="topic_2",
        question="What is Playwright?",
        rationale="r",
        search_terms=["playwright testing framework"],
    )

    ranked = rank_topics_for_source(
        [topic_1, topic_2],
        text="Playwright is a testing framework maintained by Microsoft.",
    )

    assert ranked[0] == "topic_2"


def test_choose_active_topic_prefers_in_progress_topic_for_depth() -> None:
    pending = ResearchTopic(
        id="topic_pending",
        question="What is Playwright?",
        rationale="Need baseline",
        status=TopicStatus.PENDING,
        search_terms=["playwright"],
    )
    in_progress = ResearchTopic(
        id="topic_focus",
        question="What are Tavily search capabilities?",
        rationale="Need comparative depth",
        status=TopicStatus.IN_PROGRESS,
        search_terms=["tavily search api"],
    )

    chosen = choose_active_topic(
        [pending, in_progress],
        {"topic_pending": 0, "topic_focus": 1},
        {
            "topic_pending": TopicCoverage(topic_id="topic_pending"),
            "topic_focus": TopicCoverage(topic_id="topic_focus", accepted_evidence_count=1, attempts=1),
        },
    )

    assert chosen is not None
    assert chosen.id == "topic_focus"


def test_validate_candidate_rejects_feed_with_weak_match() -> None:
    topic = ResearchTopic(
        id="topic_news",
        question="resume las ultimas noticias de sucesos de ayer en castellon",
        rationale="Need local incidents coverage",
        search_terms=["sucesos castellon ayer"],
    )
    candidate = SearchCandidate(
        url="https://gcdiario.com/seccion/sucesos/feed",
        title="SUCESOS archivos - GC Diario",
        snippet="Ultimos sucesos en castellano y Galicia",
        domain="gcdiario.com",
    )

    is_valid, note = validate_candidate_for_topic(candidate, topic)

    assert is_valid is False
    assert "Feed or RSS" in note


def test_validate_candidate_accepts_article_with_location_match() -> None:
    topic = ResearchTopic(
        id="topic_news",
        question="resume las ultimas noticias de sucesos de ayer en castellon",
        rationale="Need local incidents coverage",
        search_terms=["sucesos castellon ayer"],
    )
    candidate = SearchCandidate(
        url="https://castellonplaza.com/sucesos/castellon-detencion-ayer-centro",
        title="Detenido un hombre tras un altercado en Castellon",
        snippet="Sucesos de ayer en Castellon con intervencion policial en el centro.",
        domain="castellonplaza.com",
    )

    is_valid, note = validate_candidate_for_topic(candidate, topic)

    assert is_valid is True
    assert note == ""
