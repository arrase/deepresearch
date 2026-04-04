from deepresearch.core.utils import (
    canonicalize_url,
    compute_minimum_coverage,
    deduplicate_candidates,
    rank_topics_for_source,
)
from deepresearch.state import CuratedEvidence, EvidenceSourceRef, ResearchTopic, SearchCandidate


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
        question="What is Lightpanda?",
        rationale="r",
        search_terms=["lightpanda framework"],
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
