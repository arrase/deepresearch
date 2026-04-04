"""Text chunking, token estimation, excerpt helpers, and browser payload sanitization."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .url import extract_domain

_BROWSER_NOISE_TOKENS = (
    "$time=",
    "$scope=",
    "$level=",
    "$msg=",
    "window.reporterror",
    "minified react error",
    "blocked by robots",
    "robots.txt",
    "cloudflare",
    "captcha",
    "access denied",
)


def estimate_tokens(text: str) -> int:
    """Rough token estimate assuming ~4 characters per token."""
    return max(1, len(text) // 4)


def short_excerpt(text: str, limit: int = 320) -> str:
    return " ".join(text.split())[:limit].strip()


def is_browser_noise_line(line: str) -> bool:
    normalized = " ".join(line.split())
    if not normalized:
        return False
    lowered = normalized.lower()
    return any(token in lowered for token in _BROWSER_NOISE_TOKENS)


def split_browser_payload(text: str, *, max_chars: int | None = None) -> tuple[str, str]:
    clean_lines: list[str] = []
    diagnostic_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            if clean_lines and clean_lines[-1] != "":
                clean_lines.append("")
            continue
        if is_browser_noise_line(line):
            diagnostic_lines.append(" ".join(line.split())[:300])
            continue
        clean_lines.append(raw_line.rstrip())

    content = "\n".join(clean_lines).strip()
    if max_chars is not None:
        content = content[:max_chars].strip()
    diagnostics = "\n".join(dict.fromkeys(diagnostic_lines))
    return content, diagnostics


def sanitize_source_title(title: str, url: str | None = None) -> str:
    normalized = " ".join(title.split()).strip("#|- ")
    lowered = normalized.lower()
    fallback = extract_domain(url) if url else ""
    if not normalized:
        return fallback
    if any(token in lowered for token in _BROWSER_NOISE_TOKENS):
        return fallback
    if not re.search(r"[a-z0-9]", lowered):
        return fallback
    alnum_length = len(re.sub(r"[^a-z0-9]", "", lowered))
    if alnum_length < 3:
        return fallback
    return normalized[:200]


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    cleaned = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    if len(cleaned) <= chunk_size:
        return [cleaned] if cleaned else []
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def select_relevant_chunks(chunks: Iterable[str], query_terms: Iterable[str], max_chunks: int = 4) -> list[str]:
    terms = {t.lower() for t in query_terms if t.strip()}
    if not terms:
        return list(chunks)[:max_chunks]
    scored = sorted(
        ((sum(1 for t in terms if t in c.lower()), c) for c in chunks),
        key=lambda x: x[0],
        reverse=True,
    )
    return [c for s, c in scored[:max_chunks] if c.strip()]
