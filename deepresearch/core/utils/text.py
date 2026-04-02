"""Text chunking, token estimation, and excerpt helpers."""

from __future__ import annotations

from collections.abc import Iterable


def estimate_tokens(text: str) -> int:
    """Rough token estimate assuming ~4 characters per token."""
    return max(1, len(text) // 4)


def short_excerpt(text: str, limit: int = 320) -> str:
    return " ".join(text.split())[:limit].strip()


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
