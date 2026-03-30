from state import AtomicEvidence, Subquery
from subagents.deterministic import canonicalize_url, compute_minimum_coverage


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
