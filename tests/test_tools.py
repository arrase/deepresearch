from __future__ import annotations

from typing import Any

from deepresearch.config import ResearchConfig
from deepresearch.core.utils import classify_browser_payload, sanitize_source_title, split_browser_payload
from deepresearch.core.utils.evidence import build_report_sources
from deepresearch.state import BrowserPageStatus, CuratedEvidence, EvidenceSourceRef
from deepresearch.tools import DuckDuckGoSearchClient


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None


def test_duckduckgo_lite_parser_extracts_candidates(monkeypatch) -> None:
    config = ResearchConfig()
    client = DuckDuckGoSearchClient(config.search)
    sample_html = """
    <html>
      <body>
        <table>
          <tr><td><a class="result-link"
            href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Freport">Example result</a></td></tr>
          <tr><td>This is the snippet for the example result.</td></tr>
          <tr><td>example.com/report</td></tr>
        </table>
      </body>
    </html>
    """
    monkeypatch.setattr(
        client._client,
        "get",
        lambda *args, **kwargs: FakeResponse(sample_html),
    )

    results = client.search("example")

    assert len(results) == 1
    assert results[0].url == "https://example.com/report"
    assert results[0].title == "Example result"
    assert "snippet" in results[0].snippet


def test_duckduckgo_lite_retries_with_normalized_query_after_anomaly(monkeypatch) -> None:
    config = ResearchConfig()
    client = DuckDuckGoSearchClient(config.search)
    anomaly_html = "<html><body><form id='challenge-form' action='//duckduckgo.com/anomaly.js'></form></body></html>"
    sample_html = """
    <html>
      <body>
        <table>
          <tr><td><a class="result-link"
            href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fstable">Stable result</a></td></tr>
          <tr><td>Stable snippet.</td></tr>
          <tr><td>example.com/stable</td></tr>
        </table>
      </body>
    </html>
    """
    calls: list[str] = []

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        calls.append(kwargs["params"]["q"])
        if len(calls) == 1:
            return FakeResponse(anomaly_html, status_code=202)
        return FakeResponse(sample_html)

    monkeypatch.setattr(client._client, "get", fake_get)

    results = client.search("¿Qué modelos de ingresos actuales y proyectados sostienen el negocio?")

    assert len(results) == 1
    assert calls[0].startswith("¿Qué modelos")
    assert calls[1] == "Que modelos de ingresos actuales y proyectados sostienen el negocio"


def test_split_browser_payload_blocks_robot_warning_without_content() -> None:
  raw = (
    '$time=1775261612956 $scope=http $level=warn $msg="blocked by robots" '
    "url=https://www.linkedin.com/posts/lightpanda"
  )

  content, diagnostics = split_browser_payload(raw, max_chars=4000)
  status = classify_browser_payload(
    content=content,
    error=diagnostics,
    exit_code=0,
    min_partial_chars=120,
    min_useful_chars=300,
  )

  assert content == ""
  assert "blocked by robots" in diagnostics
  assert status == BrowserPageStatus.BLOCKED


def test_sanitize_source_title_falls_back_for_noise() -> None:
  title = sanitize_source_title(
    '$time=1775261744854 $scope=js $level=warn $msg=window.reportError message="Error: Minified React error #418"',
    "https://lightpanda.io/blog/posts/from-local-to-real-world-benchmarks",
  )

  assert title == "lightpanda.io"


def test_build_report_sources_prefers_clean_title_over_noise() -> None:
  noisy = CuratedEvidence(
    topic_id="topic_1",
    canonical_claim="Claim one",
    summary="Summary one",
    sources=[
      EvidenceSourceRef(
        url="https://lightpanda.io/blog/posts/from-local-to-real-world-benchmarks",
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
        url="https://lightpanda.io/blog/posts/from-local-to-real-world-benchmarks",
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
