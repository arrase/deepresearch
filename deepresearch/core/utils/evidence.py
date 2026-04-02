"""Evidence deduplication, selection, dossier updates, and source building."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from hashlib import sha1

from ...state import (
    AtomicEvidence,
    Gap,
    GapSeverity,
    ReportSource,
    Subquery,
    WorkingDossier,
)
from .text import estimate_tokens


def stable_evidence_key(source_url: str, claim: str, quotation: str) -> str:
    return sha1(f"{source_url}|{claim}|{quotation}".encode()).hexdigest()[:16]


def total_evidence_tokens(evidence: Iterable[AtomicEvidence]) -> int:
    return sum(estimate_tokens(it.claim + it.summary + it.quotation) for it in evidence)


def deduplicate_evidence(
    existing: Iterable[AtomicEvidence],
    incoming: Iterable[AtomicEvidence],
) -> list[AtomicEvidence]:
    known = {stable_evidence_key(e.source_url, e.claim, e.quotation) for e in existing}
    accepted: list[AtomicEvidence] = []
    for it in incoming:
        key = stable_evidence_key(it.source_url, it.claim, it.quotation)
        if key not in known:
            known.add(key)
            accepted.append(it)
    return accepted


def select_evidence_for_context(
    evidence: Iterable[AtomicEvidence],
    *,
    subquery_ids: Iterable[str],
    budget_tokens: int,
) -> list[AtomicEvidence]:
    wanted = set(subquery_ids)
    selected: list[AtomicEvidence] = []
    consumed = 0
    for it in sorted(evidence, key=lambda x: x.relevance_score, reverse=True):
        if wanted and it.subquery_id not in wanted:
            continue
        tokens = estimate_tokens(it.claim + it.summary + it.quotation)
        if consumed + tokens > budget_tokens:
            break
        selected.append(it)
        consumed += tokens
    return selected


def update_working_dossier(
    dossier: WorkingDossier,
    evidence: Iterable[AtomicEvidence],
    source_url: str | None = None,
    source_title: str | None = None,
) -> WorkingDossier:
    merged = dossier.model_copy(deep=True)
    for it in evidence:
        cur = merged.subquery_summaries.get(it.subquery_id, "")
        merged.subquery_summaries[it.subquery_id] = f"{cur}\n- {it.claim}".strip()
        point = f"{it.source_title}: {it.summary}"
        if point not in merged.key_points:
            merged.key_points.append(point)
    if source_url and source_title:
        merged.source_summaries[source_url] = (
            f"{source_title}: {' | '.join(e.claim for e in evidence)[:400]}"
        )
    return merged


def compute_minimum_coverage(
    active_subqueries: list[Subquery],
    evidence: Iterable[AtomicEvidence],
) -> tuple[list[str], list[Gap]]:
    counts = Counter(e.subquery_id for e in evidence)
    resolved: list[str] = []
    gaps: list[Gap] = []
    for sq in active_subqueries:
        if counts[sq.id] >= sq.evidence_target:
            resolved.append(sq.id)
        else:
            gaps.append(
                Gap(
                    subquery_id=sq.id,
                    description=f"Need {sq.evidence_target - counts[sq.id]} more evidence items.",
                    severity=GapSeverity.MEDIUM,
                )
            )
    return resolved, gaps


def build_report_sources(evidence: Iterable[AtomicEvidence]) -> list[ReportSource]:
    sources: dict[str, dict] = {}
    for e in evidence:
        s = sources.setdefault(e.source_url, {"title": e.source_title, "ids": []})
        if e.id not in s["ids"]:
            s["ids"].append(e.id)
    return [ReportSource(url=u, title=d["title"], evidence_ids=d["ids"]) for u, d in sources.items()]
