"""Topic and source selection helpers for the SLM pipeline."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from urllib.parse import urlparse

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

_LOW_VALUE_DOMAINS = {"slideshare.net"}
_SOCIAL_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "reddit.com",
    "tiktok.com",
    "x.com",
}
_LOW_SIGNAL_PATH_SEGMENTS = {
    "archive",
    "archivo",
    "archives",
    "category",
    "feed",
    "live",
    "live-news",
    "rss",
    "section",
    "seccion",
    "slideshow",
    "tag",
}


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def _tokenize_for_match(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _fold_text(text))
        if len(token) > 2
    }


def _topic_terms(topic: ResearchTopic) -> set[str]:
    return _tokenize_for_match(" ".join([topic.question, *topic.search_terms]))


def _candidate_terms(candidate: SearchCandidate) -> tuple[set[str], set[str], set[str]]:
    title_terms = _tokenize_for_match(candidate.title)
    snippet_terms = _tokenize_for_match(candidate.snippet)
    path_terms = _tokenize_for_match(candidate.url)
    return title_terms, snippet_terms, path_terms


def validate_candidate_for_topic(candidate: SearchCandidate, topic: ResearchTopic) -> tuple[bool, str]:
    topic_terms = _topic_terms(topic)
    if not topic_terms:
        return True, ""

    title_terms, snippet_terms, path_terms = _candidate_terms(candidate)
    all_candidate_terms = title_terms | snippet_terms | path_terms
    overlap = topic_terms & all_candidate_terms
    overlap_ratio = len(overlap) / len(topic_terms) if topic_terms else 0.0
    domain = _fold_text(candidate.domain or extract_domain(candidate.url))
    path_segments = {
        segment
        for segment in re.split(r"[^a-z0-9]+", _fold_text(urlparse(candidate.url).path))
        if segment
    }

    if domain in _LOW_VALUE_DOMAINS:
        return False, f"Low-value domain for research: {domain}"
    if domain in _SOCIAL_DOMAINS and overlap_ratio < 0.3:
        return False, f"Social source with weak topical match: {domain}"
    if "feed" in path_segments or "rss" in path_segments:
        return False, "Feed or RSS endpoint without article-level context"
    if path_segments & _LOW_SIGNAL_PATH_SEGMENTS and overlap_ratio < 0.3:
        return False, "Listing page with weak topical overlap"
    if overlap_ratio < 0.15:
        return False, "No strong topical overlap with the active topic"
    return True, ""


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
            topic.status != TopicStatus.IN_PROGRESS,
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
    score += min(len(candidate.snippet) / 180.0, 1.0)
    title_terms, snippet_terms, path_terms = _candidate_terms(candidate)
    topic_terms = _topic_terms(topic)
    norm = max(len(topic_terms), 1)
    score += 1.5 * len(topic_terms & title_terms) / norm
    score += 0.75 * len(topic_terms & snippet_terms) / norm
    score += 1.0 * len(topic_terms & path_terms) / norm
    if domain_counts[candidate.domain] == 0:
        score += 0.3
    if candidate.url.startswith("https://"):
        score += 0.1
    # Bonus for candidates that carry Tavily raw_content
    if candidate.raw_content:
        score += 0.5
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
    error_text = (error or "").lower()
    content_text = content.lower()
    blocked_tokens = (
        "blocked by robots",
        "robots.txt",
        "access denied",
        "captcha",
        "cloudflare",
        "403",
        "forbidden",
        "429",
        "too many requests",
    )
    technical_tokens = (
        "window.reporterror",
        "minified react error",
        "timed out",
        "timeout",
        "connection reset",
        "dns",
        "browser error",
    )

    if any(token in error_text for token in blocked_tokens) or any(token in content_text for token in blocked_tokens):
        return BrowserPageStatus.BLOCKED

    if error_text:
        if any(token in error_text for token in ("404", "not found", "dns", "timed out", "timeout")):
            return BrowserPageStatus.RECOVERABLE_ERROR
        if not content.strip() and (
            exit_code not in (None, 0) or any(token in error_text for token in technical_tokens)
        ):
            return BrowserPageStatus.RECOVERABLE_ERROR

    chars = len(content.strip())
    if chars >= min_useful_chars:
        return BrowserPageStatus.USEFUL
    if chars >= min_partial_chars:
        return BrowserPageStatus.PARTIAL
    if error_text:
        return BrowserPageStatus.RECOVERABLE_ERROR
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
