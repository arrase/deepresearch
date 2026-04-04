"""Topic and source selection helpers for the SLM pipeline."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from ...state import (
    BrowserPageStatus,
    CuratedEvidence,
    Gap,
    ResearchTopic,
    SearchCandidate,
    SourceDiscardReason,
    SourceRecord,
    TopicCoverage,
    TopicStatus,
)
from .url import canonicalize_url, extract_domain


def choose_active_topic(
    plan: list[ResearchTopic],
    topic_attempts: dict[str, int],
    topic_coverage: dict[str, TopicCoverage],
) -> ResearchTopic | None:
    candidates = [topic for topic in plan if topic.status in {TopicStatus.PENDING, TopicStatus.IN_PROGRESS}]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda topic: (
            topic.status != TopicStatus.PENDING,
            topic_attempts.get(topic.id, 0),
            -(topic_coverage[topic.id].accepted_evidence_count if topic.id in topic_coverage else 0),
            topic.priority,
            topic.id,
        ),
    )[0]


def build_search_query(topic: ResearchTopic) -> str:
    terms = [term.strip() for term in topic.search_terms[:3] if term.strip()]
    if terms:
        return " ".join(dict.fromkeys([topic.question.strip(), *terms]))
    return topic.question.strip()


def score_candidate(
    candidate: SearchCandidate,
    topic: ResearchTopic,
    visited_urls: dict[str, SourceRecord],
    domain_counts: Counter[str],
) -> SearchCandidate:
    score = candidate.score
    if candidate.normalized_url in visited_urls or candidate.url in visited_urls:
        score -= 1.0
    haystack = f"{candidate.title} {candidate.snippet}".lower()
    score += min(len(candidate.snippet) / 180.0, 1.0)
    topic_terms = topic.search_terms or [topic.question]
    score += sum(1 for term in topic_terms[:6] if term.lower() in haystack)
    if domain_counts[candidate.domain] == 0:
        score += 0.3
    if candidate.url.startswith("https://"):
        score += 0.1
    candidate.score = round(score, 4)
    return candidate


def deduplicate_candidates(
    candidates: Iterable[SearchCandidate],
    visited_urls: dict[str, SourceRecord],
) -> tuple[list[SearchCandidate], list[tuple[str, SourceDiscardReason, str]], int]:
    unique: dict[str, SearchCandidate] = {}
    discarded: list[tuple[str, SourceDiscardReason, str]] = []
    repeated_count = 0
    for candidate in candidates:
        normalized = canonicalize_url(candidate.normalized_url or candidate.url)
        candidate.url = normalized
        candidate.normalized_url = normalized
        candidate.domain = extract_domain(normalized)
        if normalized in visited_urls:
            repeated_count += 1
            discarded.append((normalized, SourceDiscardReason.ALREADY_VISITED, "Already visited"))
            continue
        if normalized in unique:
            repeated_count += 1
            existing = unique[normalized]
            existing.topic_ids = list(dict.fromkeys([*existing.topic_ids, *candidate.topic_ids]))
            if len(candidate.snippet) > len(existing.snippet):
                existing.snippet = candidate.snippet
            if candidate.score > existing.score:
                existing.score = candidate.score
            continue
        unique[normalized] = candidate
    return list(unique.values()), discarded, repeated_count


def rank_topics_for_source(
    topics: Iterable[ResearchTopic],
    *,
    text: str,
    limit: int = 3,
) -> list[str]:
    lowered = text.lower()
    text_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    ranked: list[tuple[int, int, str]] = []
    for topic in topics:
        score = 0
        for term in (topic.search_terms or [topic.question])[:6]:
            normalized = term.strip().lower()
            if not normalized:
                continue
            if normalized in lowered:
                score += 2
                continue
            score += len({token for token in re.findall(r"[a-z0-9_]+", normalized) if len(token) > 2} & text_tokens)
        if score > 0:
            ranked.append((score, topic.priority, topic.id))
    ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [topic_id for _, _, topic_id in ranked[:limit]]


def classify_browser_payload(
    *,
    content: str,
    error: str | None,
    exit_code: int | None,
    min_partial_chars: int,
    min_useful_chars: int,
) -> BrowserPageStatus:
    if error:
        if any(token in error.lower() for token in ("404", "not found", "dns", "timed out")):
            return BrowserPageStatus.RECOVERABLE_ERROR
        return BrowserPageStatus.TERMINAL_ERROR
    if any(token in content.lower() for token in ("captcha", "access denied", "cloudflare")):
        return BrowserPageStatus.BLOCKED
    chars = len(content.strip())
    if chars >= min_useful_chars:
        return BrowserPageStatus.USEFUL
    if chars >= min_partial_chars:
        return BrowserPageStatus.PARTIAL
    return BrowserPageStatus.EMPTY


def enrich_gaps_with_search_terms(gaps: list[Gap], plan: list[ResearchTopic]) -> list[Gap]:
    topic_map = {topic.id: topic for topic in plan}
    enriched: list[Gap] = []
    for gap in gaps:
        if gap.suggested_queries:
            enriched.append(gap)
            continue
        topic = topic_map.get(gap.topic_id)
        if topic is None:
            enriched.append(gap)
            continue
        enriched.append(
            gap.model_copy(update={"suggested_queries": list(topic.search_terms[:3]) or [topic.question]})
        )
    return enriched


def reformulate_queries(
    original_query: str,
    completed_queries: set[str],
    search_terms: list[str],
) -> list[str]:
    variants: list[str] = []
    seen = {query.lower().strip() for query in completed_queries}
    for term in search_terms[:5]:
        normalized_term = term.strip()
        if not normalized_term:
            continue
        if normalized_term.lower() not in seen:
            variants.append(normalized_term)
        combined = f"{original_query} {normalized_term}".strip()
        if combined.lower() not in seen and combined != normalized_term:
            variants.append(combined)
        words = normalized_term.split()
        if len(words) > 4:
            shorter = " ".join(words[:4])
            if shorter.lower() not in seen:
                variants.append(shorter)
    return list(dict.fromkeys(variants))[:4]


def prune_queue_by_domain(
    queue: list[SearchCandidate],
    evidence: Iterable[CuratedEvidence],
    *,
    max_per_domain: int = 2,
) -> tuple[list[SearchCandidate], list[tuple[str, SourceDiscardReason, str]]]:
    evidence_domains = Counter(extract_domain(source.url) for item in evidence for source in item.sources)
    saturated = {domain for domain, count in evidence_domains.items() if count >= 3}
    kept: list[SearchCandidate] = []
    pruned: list[tuple[str, SourceDiscardReason, str]] = []
    domain_budget: Counter[str] = Counter()
    for candidate in queue:
        if candidate.domain in saturated and domain_budget[candidate.domain] >= max_per_domain:
            pruned.append((candidate.url, SourceDiscardReason.LOW_VALUE, f"Domain {candidate.domain} over-represented"))
            continue
        kept.append(candidate)
        domain_budget[candidate.domain] += 1
    return kept, pruned
