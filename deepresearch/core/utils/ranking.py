"""Candidate scoring, deduplication, and queue management."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from ...state import (
    AtomicEvidence,
    BrowserPageStatus,
    Gap,
    SearchCandidate,
    SourceDiscardReason,
    Subquery,
)
from .url import canonicalize_url, extract_domain


def score_candidate(
    candidate: SearchCandidate,
    active_subqueries: list[Subquery],
    visited_urls: dict,
    domain_counts: Counter,
) -> SearchCandidate:
    score = candidate.score
    if candidate.url in visited_urls:
        score -= 1.0
    if candidate.snippet:
        score += min(len(candidate.snippet) / 180.0, 1.0)

    haystack = f"{candidate.title} {candidate.snippet}".lower()
    matches = sum(
        1
        for sq in active_subqueries
        if any(t.lower() in haystack for t in (sq.search_terms or [sq.question])[:6])
    )
    score += 1.25 * matches

    if domain_counts[candidate.domain] == 0:
        score += 0.3
    if candidate.url.startswith("https://"):
        score += 0.1

    candidate.score = round(score, 4)
    return candidate


def deduplicate_candidates(
    candidates: Iterable[SearchCandidate],
    visited_urls: dict,
) -> tuple[list[SearchCandidate], list[tuple]]:
    unique: dict[str, SearchCandidate] = {}
    discarded: list[tuple] = []
    for c in candidates:
        url = canonicalize_url(c.url)
        c.url, c.domain = url, extract_domain(url)
        if url in visited_urls:
            discarded.append((url, SourceDiscardReason.ALREADY_VISITED, "Already visited"))
        elif url in unique:
            existing = unique[url]
            existing.subquery_ids = list(dict.fromkeys([*existing.subquery_ids, *c.subquery_ids]))
            existing.reasons = list(dict.fromkeys([*existing.reasons, *c.reasons]))
            if len(c.snippet) > len(existing.snippet):
                existing.snippet = c.snippet
            if not existing.title and c.title:
                existing.title = c.title
            existing.score = max(existing.score, c.score)
        else:
            unique[url] = c
    return list(unique.values()), discarded


def rank_subqueries_for_source(
    active_subqueries: list[Subquery],
    *,
    text: str,
    candidate_subquery_ids: Iterable[str] | None = None,
    limit: int = 3,
) -> list[str]:
    lowered = text.lower()
    text_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    allowed_ids = set(candidate_subquery_ids or [])
    ranked: list[tuple[int, int, str]] = []

    for subquery in active_subqueries:
        if allowed_ids and subquery.id not in allowed_ids:
            continue

        score = 0
        for term in (subquery.search_terms or [subquery.question])[:6]:
            normalized = term.strip().lower()
            if normalized and normalized in lowered:
                score += 2 if " " in normalized else 1
                continue
            term_tokens = {token for token in re.findall(r"[a-z0-9_]+", normalized) if len(token) > 2}
            score += len(term_tokens & text_tokens)
        question = subquery.question.strip().lower()
        if question and question in lowered:
            score += 2
        else:
            question_tokens = {token for token in re.findall(r"[a-z0-9_]+", question) if len(token) > 2}
            score += len(question_tokens & text_tokens)

        if score > 0 or (allowed_ids and subquery.id in allowed_ids):
            ranked.append((score, subquery.priority, subquery.id))

    if ranked:
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [subquery_id for _, _, subquery_id in ranked[:limit]]

    if allowed_ids:
        return [subquery.id for subquery in active_subqueries if subquery.id in allowed_ids][:limit]

    return []


def classify_browser_payload(
    *,
    content: str,
    error: str | None,
    exit_code: int | None,
    min_partial_chars: int,
    min_useful_chars: int,
) -> BrowserPageStatus:
    if error:
        if any(t in error.lower() for t in ("404", "not found", "dns", "timed out")):
            return BrowserPageStatus.RECOVERABLE_ERROR
        return BrowserPageStatus.TERMINAL_ERROR
    if any(t in content.lower() for t in ("captcha", "access denied", "cloudflare")):
        return BrowserPageStatus.BLOCKED
    chars = len(content.strip())
    if chars >= min_useful_chars:
        return BrowserPageStatus.USEFUL
    if chars >= min_partial_chars:
        return BrowserPageStatus.PARTIAL
    return BrowserPageStatus.EMPTY


def enrich_gaps_with_search_terms(
    gaps: list[Gap],
    active_subqueries: list[Subquery],
) -> list[Gap]:
    """Ensure every gap has suggested_queries by falling back to subquery search_terms."""
    sq_map = {sq.id: sq for sq in active_subqueries}
    enriched: list[Gap] = []
    for gap in gaps:
        if gap.suggested_queries:
            enriched.append(gap)
            continue
        sq = sq_map.get(gap.subquery_id)
        if sq and sq.search_terms:
            gap = gap.model_copy(update={"suggested_queries": list(sq.search_terms[:5])})
        elif sq:
            gap = gap.model_copy(update={"suggested_queries": [sq.question]})
        enriched.append(gap)
    return enriched


def reformulate_queries(
    original_query: str,
    completed_queries: set[str],
    search_terms: list[str],
) -> list[str]:
    """Generate alternative search queries when existing ones yield no results."""
    variants: list[str] = []
    seen = {q.lower().strip() for q in completed_queries}

    for term in search_terms[:5]:
        words = term.split()
        if len(words) > 7:
            short = " ".join(words[:5])
            if short.lower().strip() not in seen:
                variants.append(short)

        combined = f"{original_query} {term}"
        if combined.lower().strip() not in seen and combined != term:
            variants.append(combined)

    if not variants and search_terms:
        fallback = " ".join(search_terms[:3])
        if fallback.lower().strip() not in seen:
            variants.append(fallback)

    return variants[:4]


def prune_queue_by_domain(
    queue: list[SearchCandidate],
    evidence: Iterable[AtomicEvidence],
    *,
    max_per_domain: int = 2,
) -> tuple[list[SearchCandidate], list[tuple]]:
    """Remove candidates from over-represented domains based on existing evidence."""
    evidence_domains: Counter = Counter(extract_domain(e.source_url) for e in evidence)
    saturated = {d for d, c in evidence_domains.items() if c >= 3}
    kept: list[SearchCandidate] = []
    pruned: list[tuple] = []
    domain_budget: Counter = Counter()
    for c in queue:
        if c.domain in saturated and domain_budget[c.domain] >= max_per_domain:
            pruned.append((c.url, SourceDiscardReason.LOW_VALUE, f"Domain {c.domain} over-represented"))
        else:
            kept.append(c)
            domain_budget[c.domain] += 1
    return kept, pruned
