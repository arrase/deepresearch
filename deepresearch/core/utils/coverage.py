"""Summarization helpers for state objects and coverage rendering."""

from __future__ import annotations

from collections.abc import Iterable

from ...state import (
    AtomicEvidence,
    FinalReport,
    Gap,
    SearchCandidate,
    SourceVisit,
    Subquery,
)
from .text import short_excerpt


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


def summarize_search_candidates(
    candidates: Iterable[SearchCandidate],
    limit: int = 5,
) -> list[dict[str, object]]:
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


def summarize_source_visit(
    source: SourceVisit,
    *,
    include_content_preview: bool = False,
) -> dict[str, object]:
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
