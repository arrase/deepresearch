"""Summarization helpers for the SLM-oriented state models."""

from __future__ import annotations

from collections.abc import Iterable

from ...state import CuratedEvidence, FinalReport, Gap, ResearchTopic, SearchCandidate, SourceVisit
from .text import short_excerpt


def render_markdown_report(report: FinalReport) -> str:
    return report.markdown_report


def summarize_subqueries(topics: Iterable[ResearchTopic], limit: int = 5) -> list[dict[str, object]]:
    return [
        {
            "id": topic.id,
            "question": topic.question,
            "status": topic.status.value,
            "priority": topic.priority,
            "evidence_target": topic.evidence_target,
            "search_terms": topic.search_terms[:6],
        }
        for topic in list(topics)[:limit]
    ]


def summarize_gaps(gaps: Iterable[Gap], limit: int = 5) -> list[dict[str, object]]:
    return [
        {
            "topic_id": gap.topic_id,
            "description": gap.description,
            "severity": gap.severity.value,
            "actionable": gap.actionable,
            "suggested_queries": gap.suggested_queries[:5],
        }
        for gap in list(gaps)[:limit]
    ]


def summarize_evidence(evidence: Iterable[CuratedEvidence], limit: int = 5) -> list[dict[str, object]]:
    items = list(evidence)[:limit]
    return [
        {
            "id": item.evidence_id,
            "topic_id": item.topic_id,
            "claim": item.canonical_claim,
            "summary": item.summary,
            "support_quotes": [short_excerpt(quote, 200) for quote in item.support_quotes[:2]],
            "sources": [source.url for source in item.sources[:3]],
            "novelty_score": item.novelty_score,
            "confidence": item.confidence.value,
        }
        for item in items
    ]


def summarize_search_candidates(candidates: Iterable[SearchCandidate], limit: int = 5) -> list[dict[str, object]]:
    return [
        {
            "url": item.url,
            "title": item.title,
            "domain": item.domain,
            "score": item.score,
            "topic_ids": item.topic_ids[:6],
            "query": item.query,
        }
        for item in list(candidates)[:limit]
    ]


def summarize_source_visit(source: SourceVisit, *, include_content_preview: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "url": source.url,
        "final_url": source.final_url,
        "title": source.title,
        "status": source.status.value,
        "excerpt": short_excerpt(source.excerpt, 200),
        "content_chars": len(source.content),
        "topic_ids": source.topic_ids[:6],
        "diagnostics": source.diagnostics,
        "error": source.error,
    }
    if include_content_preview:
        payload["content_preview"] = short_excerpt(source.content, 400)
    return payload
