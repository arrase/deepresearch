"""Tavily API search backend."""

from __future__ import annotations

from types import TracebackType

import httpx

from ..config import SearchConfig
from ..core.utils import canonicalize_url, extract_domain
from ..state import SearchCandidate


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
            "include_raw_content": True,
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
            raw_content = result.get("raw_content") or ""
            candidates.append(
                SearchCandidate(
                    url=canonicalize_url(url),
                    normalized_url=canonicalize_url(url),
                    title=result.get("title", "")[:300],
                    snippet=result.get("content", "")[:500],
                    domain=extract_domain(url),
                    score=result.get("score", 0.0),
                    raw_content=raw_content[:self._config.max_raw_content_chars],
                )
            )
        return candidates

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TavilySearchClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
