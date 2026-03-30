"""Integraciones externas: Lightpanda y descubrimiento de fuentes.

Lightpanda se gestiona mediante Docker SDK desde el arranque para cumplir el
requisito operativo de descargar la imagen del navegador al iniciar la
aplicacion. El adaptador usa una ejecucion determinista del comando fetch,
clasifica resultados tecnicos y devuelve un BrowserResult estable para el grafo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

import docker
import httpx
from bs4 import BeautifulSoup
from docker.errors import DockerException

from .config import BrowserConfig, ModelConfig, SearchConfig
from .state import BrowserPageStatus, BrowserResult, SearchCandidate
from .subagents.deterministic import (
    canonicalize_url,
    classify_browser_payload,
    extract_domain,
    short_excerpt,
)


class SelfCheckError(RuntimeError):
    """Error operativo recuperable detectado durante el self-check."""


@dataclass
class SelfCheckReport:
    docker_ok: bool
    ollama_ok: bool
    model_available: bool
    lightpanda_image_ready: bool
    details: dict[str, Any]


class DuckDuckGoSearchClient:
    """Backend inicial de descubrimiento de fuentes sin API key."""

    def __init__(self, config: SearchConfig) -> None:
        self._config = config
        self._client = httpx.Client(
            timeout=20.0,
            follow_redirects=True,
            headers={"user-agent": self._config.user_agent},
        )

    def search(self, query: str, *, max_results: int | None = None) -> list[SearchCandidate]:
        target_max = max_results or self._config.results_per_query
        response = self._client.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        candidates: list[SearchCandidate] = []
        for result in soup.select("div.result"):
            anchor = result.select_one("a.result__a")
            if anchor is None:
                continue
            raw_href = anchor.get("href", "").strip()
            url = self._resolve_result_url(raw_href)
            if not url.startswith("http"):
                continue
            snippet_node = result.select_one("a.result__snippet, div.result__snippet")
            title = anchor.get_text(" ", strip=True)
            snippet = snippet_node.get_text(" ", strip=True) if snippet_node else ""
            candidates.append(
                SearchCandidate(
                    url=canonicalize_url(url),
                    title=title[:300],
                    snippet=snippet[:500],
                    domain=extract_domain(url),
                    source_type="search_result",
                )
            )
            if len(candidates) >= target_max:
                break
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


class LightpandaDockerManager:
    """Wrapper determinista sobre Lightpanda ejecutado en Docker.

    Se usa el comando fetch del contenedor oficial porque devuelve una
    representacion textual ya estabilizada por el navegador. El manager mantiene
    el contrato aislado de Docker para que el resto del sistema solo vea
    BrowserResult clasificados.
    """

    def __init__(self, config: BrowserConfig) -> None:
        self._config = config
        self._client = docker.from_env()

    @property
    def image(self) -> str:
        return self._config.image

    def ensure_image(self) -> None:
        self._client.ping()
        self._client.images.pull(self._config.image)

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


def self_check_services(
    *,
    browser: LightpandaDockerManager,
    model: ModelConfig,
) -> SelfCheckReport:
    details: dict[str, Any] = {}
    docker_ok = False
    ollama_ok = False
    model_available = False
    lightpanda_image_ready = False

    try:
        browser.ensure_image()
        docker_ok = True
        lightpanda_image_ready = True
        details["lightpanda_image"] = browser.image
    except Exception as exc:  # noqa: BLE001
        details["docker_error"] = str(exc)

    try:
        with httpx.Client(base_url=model.base_url, timeout=10.0) as client:
            tags_response = client.get("/api/tags")
            tags_response.raise_for_status()
            ollama_ok = True
            models = tags_response.json().get("models", [])
            model_names = {entry.get("name", "") for entry in models}
            model_available = model.model_name in model_names
            details["available_models"] = sorted(name for name in model_names if name)
    except Exception as exc:  # noqa: BLE001
        details["ollama_error"] = str(exc)

    return SelfCheckReport(
        docker_ok=docker_ok,
        ollama_ok=ollama_ok,
        model_available=model_available,
        lightpanda_image_ready=lightpanda_image_ready,
        details=details,
    )
