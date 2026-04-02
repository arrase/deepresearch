"""Docker-managed Lightpanda browser adapter."""

from __future__ import annotations

from urllib.parse import urlparse

import docker

from ..config import BrowserConfig
from ..core.utils import classify_browser_payload, short_excerpt
from ..state import BrowserPageStatus, SourceVisit


class LightpandaDockerManager:
    """Deterministic wrapper around Lightpanda running in Docker."""

    def __init__(self, config: BrowserConfig) -> None:
        self._config = config
        self._client = docker.from_env()

    def fetch(self, url: str) -> SourceVisit:
        parsed_url = urlparse(url)
        if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
            return SourceVisit(
                url=url,
                status=BrowserPageStatus.TERMINAL_ERROR,
                error=f"Invalid URL scheme or missing host: {url}",
            )

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
            return SourceVisit(
                url=url,
                final_url=url,
                status=status,
                title=self._extract_title(content),
                content=content,
                excerpt=short_excerpt(content),
                diagnostics={
                    "image": self._config.image,
                    "wait_ms": self._config.wait_ms,
                },
            )
        except Exception as exc:  # noqa: BLE001
            return SourceVisit(
                url=url,
                status=BrowserPageStatus.TERMINAL_ERROR,
                error=str(exc),
                diagnostics={"image": self._config.image},
            )

    @staticmethod
    def _extract_title(content: str) -> str:
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or (line.startswith("[") and "](" in line) or line.startswith("-"):
                continue
            cleaned = line.lstrip("#").strip()
            if cleaned:
                return cleaned[:200]
        return ""
