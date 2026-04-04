"""Evidence curation, coverage, and dossier helpers for the SLM pipeline."""

from __future__ import annotations

import re
from collections.abc import Iterable
from hashlib import sha1

from ...config import DedupConfig
from ...state import (
    CuratedEvidence,
    EvidenceDraft,
    EvidenceSourceRef,
    Gap,
    GapSeverity,
    ReportSource,
    ResearchTopic,
    TopicCoverage,
    TopicStatus,
    WorkingDossier,
)
from .text import estimate_tokens
from .url import extract_domain

_SIGNIFICANT_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)?")
_NUMERIC_TOKEN_RE = re.compile(
    r"\b\d+(?:[\.,]\d+)?(?:\s*(?:%|km|m|cm|mm|kg|g|mg|b|kb|mb|gb|tb|kwh|mwh|gwh|twh|usd|eur))?\b"
)


def normalize_claim(claim: str, *, strip_punctuation: bool = True) -> str:
    normalized = claim.strip().lower()
    if strip_punctuation:
        normalized = re.sub(r"[^a-z0-9\s%\.\-/]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def canonical_fingerprint(claim: str, config: DedupConfig) -> str:
    if not config.lexical_fingerprint:
        return sha1(claim.encode("utf-8")).hexdigest()[:16]
    normalized = normalize_claim(claim, strip_punctuation=True)
    return sha1(normalized.encode("utf-8")).hexdigest()[:16]


def numeric_tokens(text: str) -> set[str]:
    return {match.group(0).replace(" ", "") for match in _NUMERIC_TOKEN_RE.finditer(text.lower())}


def significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in _SIGNIFICANT_TOKEN_RE.findall(normalize_claim(text, strip_punctuation=True))
        if len(token) > 2 and not token.isdigit()
    }


def claims_are_approximate_duplicates(left: str, right: str, config: DedupConfig) -> bool:
    left_numbers = numeric_tokens(left)
    right_numbers = numeric_tokens(right)
    if left_numbers != right_numbers:
        return False

    left_length = max(1, len(left))
    right_length = max(1, len(right))
    ratio = left_length / right_length
    if ratio < config.min_length_ratio or ratio > config.max_length_ratio:
        return False

    left_tokens = significant_tokens(left)
    right_tokens = significant_tokens(right)
    if not left_tokens or not right_tokens:
        return False
    jaccard = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    return jaccard >= config.approximate_jaccard_threshold


def total_evidence_tokens(evidence: Iterable[CuratedEvidence]) -> int:
    return sum(item.prompt_fit_tokens_estimate for item in evidence)


def select_evidence_for_context(
    evidence: Iterable[CuratedEvidence],
    *,
    topic_ids: Iterable[str],
    budget_tokens: int,
) -> list[CuratedEvidence]:
    wanted = {topic_id for topic_id in topic_ids if topic_id}
    selected: list[CuratedEvidence] = []
    consumed = 0
    ranked = sorted(
        evidence,
        key=lambda item: (
            item.topic_id not in wanted if wanted else False,
            -item.novelty_score,
            -item.last_confirmed_iteration,
        ),
    )
    for item in ranked:
        if wanted and item.topic_id not in wanted:
            continue
        if consumed + item.prompt_fit_tokens_estimate > budget_tokens:
            break
        selected.append(item)
        consumed += item.prompt_fit_tokens_estimate
    return selected


def _merge_source(existing: CuratedEvidence, draft: EvidenceDraft) -> bool:
    already_present = any(
        source.url == draft.source_url and source.locator == draft.locator
        for source in existing.sources
    )
    if already_present:
        return False
    existing.sources.append(
        EvidenceSourceRef(url=draft.source_url, title=draft.source_title, locator=draft.locator)
    )
    if draft.quotation and draft.quotation not in existing.support_quotes:
        existing.support_quotes.append(draft.quotation)
    if draft.id not in existing.merged_from_drafts:
        existing.merged_from_drafts.append(draft.id)
    return True


def curate_evidence(
    existing: list[CuratedEvidence],
    drafts: Iterable[EvidenceDraft],
    *,
    iteration: int,
    dedup_config: DedupConfig,
) -> tuple[list[CuratedEvidence], list[CuratedEvidence], int, int]:
    curated = [item.model_copy(deep=True) for item in existing]
    accepted: list[CuratedEvidence] = []
    merged_count = 0
    exact_added_tokens = 0

    for draft in drafts:
        fingerprint = canonical_fingerprint(draft.claim, dedup_config)
        exact_match = next(
            (
                item
                for item in curated
                if item.topic_id == draft.topic_id and item.canonical_fingerprint == fingerprint
            ),
            None,
        )
        if exact_match is not None:
            if _merge_source(exact_match, draft):
                exact_match.last_confirmed_iteration = iteration
                merged_count += 1
            continue

        approximate_match = next(
            (
                item
                for item in curated
                if item.topic_id == draft.topic_id
                and claims_are_approximate_duplicates(item.canonical_claim, draft.claim, dedup_config)
            ),
            None,
        )
        if approximate_match is not None:
            if _merge_source(approximate_match, draft):
                approximate_match.last_confirmed_iteration = iteration
                merged_count += 1
            continue

        prompt_fit_tokens = estimate_tokens(f"{draft.claim} {draft.summary} {draft.quotation}")
        curated_item = CuratedEvidence(
            topic_id=draft.topic_id,
            canonical_claim=draft.claim.strip(),
            summary=draft.summary.strip(),
            support_quotes=[draft.quotation] if draft.quotation else [],
            sources=[EvidenceSourceRef(url=draft.source_url, title=draft.source_title, locator=draft.locator)],
            confidence=draft.extraction_confidence,
            novelty_score=draft.relevance_score,
            exact_generation_tokens=draft.extractor_output_tokens,
            prompt_fit_tokens_estimate=prompt_fit_tokens,
            first_seen_iteration=iteration,
            last_confirmed_iteration=iteration,
            merged_from_drafts=[draft.id],
            canonical_fingerprint=fingerprint,
        )
        curated.append(curated_item)
        accepted.append(curated_item)
        exact_added_tokens += draft.extractor_output_tokens

    return curated, accepted, merged_count, exact_added_tokens


def update_working_dossier(
    dossier: WorkingDossier,
    evidence: Iterable[CuratedEvidence],
) -> WorkingDossier:
    merged = dossier.model_copy(deep=True)
    for item in evidence:
        current = merged.topic_summaries.get(item.topic_id, "")
        line = f"- {item.canonical_claim}"
        merged.topic_summaries[item.topic_id] = "\n".join(part for part in [current, line] if part).strip()
        point = f"{item.topic_id}: {item.summary}"
        if point not in merged.key_points:
            merged.key_points.append(point)
        if item.sources:
            merged.source_summaries[item.sources[0].url] = item.summary
    return merged


def build_report_sources(evidence: Iterable[CuratedEvidence]) -> list[ReportSource]:
    sources: dict[str, tuple[str, list[str]]] = {}
    for item in evidence:
        for source in item.sources:
            title, evidence_ids = sources.setdefault(source.url, (source.title, []))
            if item.evidence_id not in evidence_ids:
                evidence_ids.append(item.evidence_id)
            sources[source.url] = (title, evidence_ids)
    return [
        ReportSource(url=url, title=title, evidence_ids=evidence_ids)
        for url, (title, evidence_ids) in sources.items()
    ]


def compute_topic_coverages(
    plan: Iterable[ResearchTopic],
    evidence: Iterable[CuratedEvidence],
    topic_attempts: dict[str, int],
) -> dict[str, TopicCoverage]:
    evidence_list = list(evidence)
    coverage_map: dict[str, TopicCoverage] = {}
    for topic in plan:
        topic_evidence = [item for item in evidence_list if item.topic_id == topic.id]
        domains = {extract_domain(source.url) for item in topic_evidence for source in item.sources}
        accepted_count = len(topic_evidence)
        attempts = topic_attempts.get(topic.id, 0)
        resolved = accepted_count >= topic.evidence_target and topic.status == TopicStatus.COMPLETED
        exhausted = topic.status == TopicStatus.EXHAUSTED
        gap_count = max(0, topic.evidence_target - accepted_count)
        pending_gap_text = [] if gap_count == 0 else [f"Need {gap_count} more evidence item(s)."]
        coverage_map[topic.id] = TopicCoverage(
            topic_id=topic.id,
            accepted_evidence_count=accepted_count,
            unique_domains=len(domains),
            attempts=attempts,
            empty_attempts=max(0, attempts - accepted_count),
            resolved=resolved,
            exhausted=exhausted,
            rationale="coverage acceptable" if gap_count == 0 else "coverage incomplete",
            pending_gaps=pending_gap_text,
        )
    return coverage_map


def compute_minimum_coverage(
    topics: list[ResearchTopic],
    evidence: Iterable[CuratedEvidence],
    topic_attempts: dict[str, int] | None = None,
) -> tuple[list[str], list[Gap]]:
    coverage_map = compute_topic_coverages(topics, evidence, topic_attempts or {})
    resolved = [topic.id for topic in topics if coverage_map[topic.id].accepted_evidence_count >= topic.evidence_target]
    gaps: list[Gap] = []
    for topic in topics:
        coverage = coverage_map[topic.id]
        if coverage.accepted_evidence_count >= topic.evidence_target:
            continue
        missing = topic.evidence_target - coverage.accepted_evidence_count
        gaps.append(
            Gap(
                topic_id=topic.id,
                description=f"Need {missing} more evidence item(s).",
                severity=GapSeverity.MEDIUM,
                suggested_queries=list(topic.search_terms[:3]) or [topic.question],
            )
        )
    return resolved, gaps
