"""DuckDuckGo Lite HTML search backend."""

from __future__ import annotations

from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from ..config import SearchConfig
from ..core.utils import canonicalize_url, extract_domain
from ..state import SearchCandidate
from ._helpers import is_duckduckgo_anomaly_page, normalize_search_query


class DuckDuckGoSearchClient:
    """Initial source discovery backend that does not require an API key."""

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            timeout=20.0,
            follow_redirects=True,
            headers={"user-agent": self._config.user_agent},
        )

    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        target_max = max_results or self._config.results_per_query
        attempts = [query]
        normalized_query = normalize_search_query(query)
        if normalized_query and normalized_query != query:
            attempts.append(normalized_query)

        saw_anomaly = False
        for attempt_query in attempts:
            response = self._client.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": attempt_query},
            )
            response.raise_for_status()
            if is_duckduckgo_anomaly_page(response.text):
                saw_anomaly = True
                continue
            candidates = self._parse_lite_results(response.text, max_results=target_max)
            if candidates:
                return candidates

        if saw_anomaly:
            raise RuntimeError("DuckDuckGo returned an anomaly challenge instead of search results")
        return []

    def _parse_lite_results(self, html: str, *, max_results: int) -> list[SearchCandidate]:
        soup = BeautifulSoup(html, "html.parser")
        candidates: list[SearchCandidate] = []
        rows = [row for row in soup.select("tr") if row.get_text(" ", strip=True)]
        row_index = 0
        while row_index < len(rows) and len(candidates) < max_results:
            row = rows[row_index]
            anchor = row.select_one("a.result-link")
            if anchor is None:
                row_index += 1
                continue
            raw_href = anchor.get("href", "").strip()
            url = self._resolve_result_url(raw_href)
            title = anchor.get_text(" ", strip=True)
            snippet = ""
            if row_index + 1 < len(rows):
                snippet = rows[row_index + 1].get_text(" ", strip=True)
            if not url.startswith("http"):
                row_index += 1
                continue
            candidates.append(
                SearchCandidate(
                    url=canonicalize_url(url),
                    title=title[:300],
                    snippet=snippet[:500],
                    domain=extract_domain(url),
                    source_type="search_result",
                )
            )
            row_index += 3
        return candidates

    @staticmethod
    def _resolve_result_url(href: str) -> str:
        if not href:
            return ""
        absolute = urljoin("https://duckduckgo.com", href)
        parsed = urlparse(absolute)
        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return uddg[0]
        return absolute
