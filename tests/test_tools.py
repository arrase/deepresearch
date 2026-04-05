from deepresearch.core.utils import classify_source_content, sanitize_source_title, split_source_content
from deepresearch.core.utils.evidence import build_report_sources
from deepresearch.state import CuratedEvidence, EvidenceSourceRef, SourceDiscardReason


def test_split_source_content_detects_blocked_page_without_content() -> None:
    raw = (
        '$time=1775261612956 $scope=http $level=warn $msg="blocked by robots" '
        "url=https://www.linkedin.com/posts/example"
    )

    content, diagnostics = split_source_content(raw, max_chars=4000)
    reason = classify_source_content(
        content=content,
        diagnostics=diagnostics,
        min_source_chars=300,
    )

    assert content == ""
    assert "blocked by robots" in diagnostics
    assert reason == SourceDiscardReason.BLOCKED


def test_sanitize_source_title_falls_back_for_noise() -> None:
    title = sanitize_source_title(
        '$time=1775261744854 $scope=js $level=warn $msg=window.reportError message="Error: Minified React error #418"',
        "https://example.com/blog/posts/benchmarks",
    )

    assert title == "example.com"


def test_build_report_sources_prefers_clean_title_over_noise() -> None:
    noisy = CuratedEvidence(
        topic_id="topic_1",
        canonical_claim="Claim one",
        summary="Summary one",
        sources=[
            EvidenceSourceRef(
                url="https://example.com/blog/posts/benchmarks",
                title='$time=1775261744854 $scope=js $level=warn $msg=window.reportError',
                locator="p1",
            )
        ],
        prompt_fit_tokens_estimate=20,
        exact_generation_tokens=10,
    )
    clean = CuratedEvidence(
        topic_id="topic_1",
        canonical_claim="Claim two",
        summary="Summary two",
        sources=[
            EvidenceSourceRef(
                url="https://example.com/blog/posts/benchmarks",
                title="From Local To Real World Benchmarks",
                locator="p2",
            )
        ],
        prompt_fit_tokens_estimate=20,
        exact_generation_tokens=10,
    )

    sources = build_report_sources([noisy, clean])

    assert len(sources) == 1
    assert sources[0].title == "From Local To Real World Benchmarks"
