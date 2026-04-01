"""Deterministic internal workers."""

from __future__ import annotations

from collections import Counter
from hashlib import sha1
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ..state import (
    AtomicEvidence,
    BrowserPageStatus,
    FinalReport,
    Gap,
    GapSeverity,
    ReportSource,
    SearchCandidate,
    SourceDiscardReason,
    SourceVisit,
    Subquery,
    WorkingDossier,
)


TRACKING_QUERY_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"
}


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    filtered_query = [
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in TRACKING_QUERY_KEYS
    ]
    return urlunparse(parsed._replace(
        scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/") or "/", query=urlencode(filtered_query), fragment=""
    ))


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def short_excerpt(text: str, limit: int = 320) -> str:
    return " ".join(text.split())[:limit].strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def total_evidence_tokens(evidence: Iterable[AtomicEvidence]) -> int:
    return sum(estimate_tokens(it.claim + it.summary + it.quotation) for it in evidence)


def stable_evidence_key(source_url: str, claim: str, quotation: str) -> str:
    return sha1(f"{source_url}|{claim}|{quotation}".encode()).hexdigest()[:16]


def split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    cleaned = "\n".join(l.rstrip() for l in text.splitlines() if l.strip())
    if len(cleaned) <= chunk_size: return [cleaned] if cleaned else []
    chunks, start = [], 0
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned): break
        start = max(0, end - overlap)
    return chunks


def select_relevant_chunks(chunks: Iterable[str], query_terms: Iterable[str], max_chunks: int = 4) -> list[str]:
    terms = {t.lower() for t in query_terms if t.strip()}
    if not terms: return list(chunks)[:max_chunks]
    scored = sorted(((sum(1 for t in terms if t in c.lower()), c) for c in chunks), key=lambda x: x[0], reverse=True)
    return [c for s, c in scored[:max_chunks] if c.strip()]


def score_candidate(candidate: SearchCandidate, active_subqueries: list[Subquery], visited_urls: dict, domain_counts: Counter) -> SearchCandidate:
    score = candidate.score
    if candidate.url in visited_urls: score -= 1.0
    if candidate.snippet: score += min(len(candidate.snippet) / 180.0, 1.0)
    
    haystack = f"{candidate.title} {candidate.snippet}".lower()
    matches = sum(1 for sq in active_subqueries if any(t.lower() in haystack for t in (sq.search_terms or [sq.question])[:6]))
    score += 1.25 * matches
    
    if domain_counts[candidate.domain] == 0: score += 0.3
    if candidate.url.startswith("https://"): score += 0.1
    
    candidate.score = round(score, 4)
    return candidate


def deduplicate_candidates(candidates: Iterable[SearchCandidate], visited_urls: dict) -> tuple[list[SearchCandidate], list[tuple]]:
    unique, discarded = {}, []
    for c in candidates:
        url = canonicalize_url(c.url)
        c.url, c.domain = url, extract_domain(url)
        if url in visited_urls: discarded.append((url, SourceDiscardReason.ALREADY_VISITED, "Already visited"))
        elif url in unique: discarded.append((url, SourceDiscardReason.DUPLICATE_URL, "Duplicate"))
        else: unique[url] = c
    return list(unique.values()), discarded


def classify_browser_payload(*, content: str, error: str | None, exit_code: int | None, min_partial_chars: int, min_useful_chars: int) -> BrowserPageStatus:
    if error:
        if any(t in error.lower() for t in ("404", "not found", "dns", "timed out")): return BrowserPageStatus.RECOVERABLE_ERROR
        return BrowserPageStatus.TERMINAL_ERROR
    if any(t in content.lower() for t in ("captcha", "access denied", "cloudflare")): return BrowserPageStatus.BLOCKED
    chars = len(content.strip())
    if chars >= min_useful_chars: return BrowserPageStatus.USEFUL
    if chars >= min_partial_chars: return BrowserPageStatus.PARTIAL
    return BrowserPageStatus.EMPTY


def deduplicate_evidence(existing: Iterable[AtomicEvidence], incoming: Iterable[AtomicEvidence]) -> list[AtomicEvidence]:
    known = {stable_evidence_key(e.source_url, e.claim, e.quotation) for e in existing}
    accepted = []
    for it in incoming:
        key = stable_evidence_key(it.source_url, it.claim, it.quotation)
        if key not in known:
            known.add(key)
            accepted.append(it)
    return accepted


def update_working_dossier(dossier: WorkingDossier, evidence: Iterable[AtomicEvidence], source_url: str | None = None, source_title: str | None = None) -> WorkingDossier:
    merged = dossier.model_copy(deep=True)
    for it in evidence:
        cur = merged.subquery_summaries.get(it.subquery_id, "")
        merged.subquery_summaries[it.subquery_id] = f"{cur}\n- {it.claim}".strip()
        point = f"{it.source_title}: {it.summary}"
        if point not in merged.key_points: merged.key_points.append(point)
    if source_url and source_title:
        merged.source_summaries[source_url] = f"{source_title}: {' | '.join(e.claim for e in evidence)[:400]}"
    return merged


def compute_minimum_coverage(active_subqueries: list[Subquery], evidence: Iterable[AtomicEvidence]) -> tuple[list[str], list[Gap]]:
    counts = Counter(e.subquery_id for e in evidence)
    resolved, gaps = [], []
    for sq in active_subqueries:
        if counts[sq.id] >= sq.evidence_target: resolved.append(sq.id)
        else:
            gaps.append(Gap(subquery_id=sq.id, description=f"Need {sq.evidence_target - counts[sq.id]} more evidence items.", severity=GapSeverity.MEDIUM))
    return resolved, gaps


def select_evidence_for_context(evidence: Iterable[AtomicEvidence], *, subquery_ids: Iterable[str], budget_tokens: int) -> list[AtomicEvidence]:
    wanted, selected, consumed = set(subquery_ids), [], 0
    for it in sorted(evidence, key=lambda x: x.relevance_score, reverse=True):
        if wanted and it.subquery_id not in wanted: continue
        tokens = estimate_tokens(it.claim + it.summary + it.quotation)
        if consumed + tokens > budget_tokens: break
        selected.append(it)
        consumed += tokens
    return selected


def build_report_sources(evidence: Iterable[AtomicEvidence]) -> list[ReportSource]:
    sources = {}
    for e in evidence:
        s = sources.setdefault(e.source_url, {"title": e.source_title, "ids": []})
        if e.id not in s["ids"]: s["ids"].append(e.id)
    return [ReportSource(url=u, title=d["title"], evidence_ids=d["ids"]) for u, d in sources.items()]


def render_markdown_report(report: FinalReport) -> str:
    return report.markdown_report


def summarize_subqueries(subqueries: Iterable[Subquery], limit: int = 5) -> list[dict[str, object]]:
    items = list(subqueries)[:limit]
    return [
        {
            "id": sq.id,
            "question": sq.question,
            "priority": sq.priority,
            "evidence_target": sq.evidence_target,
            "search_terms": sq.search_terms[:6],
        }
        for sq in items
    ]


def summarize_gaps(gaps: Iterable[Gap], limit: int = 5) -> list[dict[str, object]]:
    items = list(gaps)[:limit]
    return [
        {
            "subquery_id": gap.subquery_id,
            "description": gap.description,
            "severity": gap.severity.value,
            "actionable": gap.actionable,
            "suggested_queries": gap.suggested_queries[:5],
        }
        for gap in items
    ]


def summarize_evidence(evidence: Iterable[AtomicEvidence], limit: int = 5) -> list[dict[str, object]]:
    items = list(evidence)[:limit]
    return [
        {
            "id": item.id,
            "subquery_id": item.subquery_id,
            "claim": item.claim,
            "quotation": short_excerpt(item.quotation, 200),
            "relevance_score": item.relevance_score,
            "confidence": item.confidence.value,
            "source_title": item.source_title,
            "source_url": item.source_url,
        }
        for item in items
    ]


def summarize_search_candidates(candidates: Iterable[SearchCandidate], limit: int = 5) -> list[dict[str, object]]:
    items = list(candidates)[:limit]
    return [
        {
            "url": item.url,
            "title": item.title,
            "domain": item.domain,
            "score": item.score,
            "reasons": item.reasons[:6],
            "subquery_ids": item.subquery_ids[:6],
        }
        for item in items
    ]


def summarize_source_visit(source: SourceVisit, *, include_content_preview: bool = False) -> dict[str, object]:
    summary: dict[str, object] = {
        "url": source.url,
        "final_url": source.final_url,
        "title": source.title,
        "status": source.status.value,
        "excerpt": source.excerpt,
        "content_chars": len(source.content),
        "candidate_subquery_ids": source.candidate_subquery_ids[:6],
        "diagnostics": source.diagnostics,
        "error": source.error,
    }
    if include_content_preview:
        summary["content_preview"] = short_excerpt(source.content, 400)
    return summary
