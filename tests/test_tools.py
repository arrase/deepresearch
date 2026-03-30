from __future__ import annotations

from deepresearch.config import ResearchConfig
from deepresearch.tools import DuckDuckGoSearchClient


class FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None


def test_duckduckgo_lite_parser_extracts_candidates() -> None:
    config = ResearchConfig()
    client = DuckDuckGoSearchClient(config.search)
    sample_html = """
    <html>
      <body>
        <table>
          <tr><td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Freport">Example result</a></td></tr>
          <tr><td>This is the snippet for the example result.</td></tr>
          <tr><td>example.com/report</td></tr>
        </table>
      </body>
    </html>
    """
    client._client.get = lambda *args, **kwargs: FakeResponse(sample_html)

    results = client.search("example")

    assert len(results) == 1
    assert results[0].url == "https://example.com/report"
    assert results[0].title == "Example result"
    assert "snippet" in results[0].snippet


def test_duckduckgo_lite_retries_with_normalized_query_after_anomaly() -> None:
    config = ResearchConfig()
    client = DuckDuckGoSearchClient(config.search)
    anomaly_html = "<html><body><form id='challenge-form' action='//duckduckgo.com/anomaly.js'></form></body></html>"
    sample_html = """
    <html>
      <body>
        <table>
          <tr><td><a class="result-link" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fstable">Stable result</a></td></tr>
          <tr><td>Stable snippet.</td></tr>
          <tr><td>example.com/stable</td></tr>
        </table>
      </body>
    </html>
    """
    calls: list[str] = []

    def fake_get(*args, **kwargs):
        calls.append(kwargs["params"]["q"])
        if len(calls) == 1:
            return FakeResponse(anomaly_html, status_code=202)
        return FakeResponse(sample_html)

    client._client.get = fake_get

    results = client.search("¿Qué modelos de ingresos actuales y proyectados sostienen el negocio?")

    assert len(results) == 1
    assert calls[0].startswith("¿Qué modelos")
    assert calls[1] == "Que modelos de ingresos actuales y proyectados sostienen el negocio"