"""External integrations: Lightpanda and web-source discovery.

Lightpanda is managed through the Docker SDK so the browser image can be
bootstrapped at startup. The adapter runs a deterministic fetch command,
classifies technical outcomes, and returns a stable BrowserResult to the graph.
"""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import parse_qs, urljoin, urlparse

import docker
import httpx
from bs4 import BeautifulSoup
from docker.errors import DockerException

from .config import BrowserConfig, SearchConfig
from .state import BrowserPageStatus, BrowserResult, SearchCandidate
from .subagents.deterministic import (
    canonicalize_url,
    classify_browser_payload,
    extract_domain,
    short_excerpt,
)


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
        normalized_query = _normalize_search_query(query)
        if normalized_query and normalized_query != query:
            attempts.append(normalized_query)

        saw_anomaly = False
        for attempt_query in attempts:
            response = self._client.get(
                "https://lite.duckduckgo.com/lite/",
                params={"q": attempt_query},
            )
            response.raise_for_status()
            if _is_duckduckgo_anomaly_page(response.text):
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

    def _resolve_result_url(self, href: str) -> str:
        if not href:
            return ""
        absolute = urljoin("https://duckduckgo.com", href)
        parsed = urlparse(absolute)
        query = parse_qs(parsed.query)
        uddg = query.get("uddg")
        if uddg:
            return uddg[0]
        return absolute


class TavilySearchClient:
    """Search backend using the Tavily API for better research results."""

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._api_key = config.api_key
        if not self._api_key:
            raise ValueError("Tavily search requires an api_key in SearchConfig")
        self._client = httpx.Client(
            base_url="https://api.tavily.com",
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )

    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        target_max = max_results or self._config.results_per_query
        payload = {
            "api_key": self._api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "include_images": False,
            "include_raw_content": False,
            "max_results": target_max,
        }
        response = self._client.post("/search", json=payload)
        response.raise_for_status()
        data = response.json()

        candidates = []
        for result in data.get("results", []):
            url = result.get("url", "")
            if not url:
                continue
            candidates.append(
                SearchCandidate(
                    url=canonicalize_url(url),
                    title=result.get("title", "")[:300],
                    snippet=result.get("content", "")[:500],
                    domain=extract_domain(url),
                    source_type="search_result",
                    score=result.get("score", 0.0),
                )
            )
        return candidates


def _is_duckduckgo_anomaly_page(content: str) -> bool:
    lowered = content.lower()
    return "anomaly.js" in lowered or "challenge-form" in lowered or "botnet" in lowered


def _normalize_search_query(query: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", query).encode("ascii", "ignore").decode("ascii")
    ascii_text = re.sub(r"[^A-Za-z0-9\s.-]", " ", ascii_text)
    ascii_text = re.sub(r"\s+", " ", ascii_text).strip()
    if not ascii_text:
        return ""
    terms = ascii_text.split()
    return " ".join(terms[:12])


class LightpandaDockerManager:
    """Deterministic wrapper around Lightpanda running in Docker.

    The official container fetch command is used because it returns text already
    stabilized by the browser. The manager isolates Docker so the rest of the
    system only sees classified BrowserResult objects.
    """

    def __init__(self, config: BrowserConfig) -> None:
        self._config = config
        self._client = docker.from_env()

    @property
    def image(self) -> str:
        return self._config.image

    def fetch(self, url: str) -> BrowserResult:
        command = [
            "/bin/lightpanda",
            "fetch",
            "--dump",
            "markdown",
            "--wait-until",
            self._config.wait_until,
            "--wait-ms",
            str(self._config.wait_ms),
        ]
        if self._config.obey_robots:
            command.append("--obey-robots")
        command.append(url)
        try:
            raw = self._client.containers.run(
                self._config.image,
                command=command,
                remove=True,
                detach=False,
                stdout=True,
                stderr=True,
                environment={
                    "LIGHTPANDA_DISABLE_TELEMETRY": "true" if self._config.disable_telemetry else "false"
                },
            )
            output = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            content = output[: self._config.max_content_chars].strip()
            status = classify_browser_payload(
                content=content,
                error=None,
                exit_code=0,
                min_partial_chars=self._config.min_partial_chars,
                min_useful_chars=self._config.min_useful_chars,
            )
            title = self._extract_title(content)
            return BrowserResult(
                url=url,
                final_url=url,
                status=status,
                title=title,
                content=content,
                excerpt=short_excerpt(content),
                exit_code=0,
                diagnostics={
                    "image": self._config.image,
                    "wait_ms": self._config.wait_ms,
                    "wait_until": self._config.wait_until,
                },
            )
        except DockerException as exc:
            return BrowserResult(
                url=url,
                status=BrowserPageStatus.RECOVERABLE_ERROR,
                error=str(exc),
                diagnostics={"image": self._config.image},
            )
        except Exception as exc:  # noqa: BLE001
            return BrowserResult(
                url=url,
                status=BrowserPageStatus.TERMINAL_ERROR,
                error=str(exc),
                diagnostics={"image": self._config.image},
            )

    def _extract_title(self, content: str) -> str:
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and "](" in line:
                continue
            if line.startswith("-"):
                continue
            cleaned = line.lstrip("#").strip()
            if cleaned:
                return cleaned[:200]
        first_line = next((line.strip("# ").strip() for line in content.splitlines() if line.strip()), "")
        return first_line[:200]
