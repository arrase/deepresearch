from deepresearch.state import AtomicEvidence, SearchCandidate, Subquery
from deepresearch.core.utils import canonicalize_url, compute_minimum_coverage, deduplicate_candidates, rank_subqueries_for_source


def test_canonicalize_url_removes_tracking_query_and_fragment() -> None:
    url = "https://Example.com/path/?utm_source=x&id=42#fragment"
    assert canonicalize_url(url) == "https://example.com/path?id=42"


def test_compute_minimum_coverage_marks_subquery_resolved() -> None:
    subquery = Subquery(question="What is X?", rationale="Need definition", evidence_target=1)
    evidence = [
        AtomicEvidence(
            subquery_id=subquery.id,
            source_url="https://example.com",
            source_title="Example",
            summary="X is defined",
            claim="X is a defined concept",
            quotation="X means ...",
            citation_locator="p1",
        )
    ]
    resolved, gaps = compute_minimum_coverage(active_subqueries=[subquery], evidence=evidence)
    assert resolved == [subquery.id]
    assert gaps == []


def test_deduplicate_candidates_merges_subquery_ids() -> None:
    candidates, discarded = deduplicate_candidates(
        [
            SearchCandidate(url="https://example.com/path", title="Example", subquery_ids=["sq_1"], reasons=["q1"]),
            SearchCandidate(url="https://example.com/path?utm_source=test", title="Example", subquery_ids=["sq_2"], reasons=["q2"]),
        ],
        visited_urls={},
    )

    assert len(candidates) == 1
    assert candidates[0].subquery_ids == ["sq_1", "sq_2"]
    assert candidates[0].reasons == ["q1", "q2"]
    assert discarded == []


def test_rank_subqueries_for_source_prefers_matching_allowed_ids() -> None:
    sq_1 = Subquery(id="sq_1", question="What is Lightpanda?", rationale="r", search_terms=["lightpanda framework"])
    sq_2 = Subquery(id="sq_2", question="What is Playwright?", rationale="r", search_terms=["playwright testing framework"])

    ranked = rank_subqueries_for_source(
        [sq_1, sq_2],
        text="Playwright is a testing framework maintained by Microsoft.",
        candidate_subquery_ids=["sq_1", "sq_2"],
    )

    assert ranked[0] == "sq_2"
