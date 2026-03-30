"""Workers internos deterministas.

Aqui viven las tareas que no requieren inferencia abierta: canonizacion,
deduplicacion, scoring heuristico, seleccion de contexto y reglas minimas de
coverage. Esto mantiene fuera del LLM todo lo que puede resolverse con codigo
normal, mas fiable y mas auditable.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha1
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from state import (
    AtomicEvidence,
    BrowserPageStatus,
    FinalReport,
    Gap,
    GapSeverity,
    ReportSource,
    SearchCandidate,
    SourceDiscardReason,
    Subquery,
    WorkingDossier,
)


TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}


def canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_KEYS
    ]
    normalized_path = parsed.path.rstrip("/") or "/"
    canonical = parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=normalized_path,
        query=urlencode(filtered_query),
        fragment="",
    )
    return urlunparse(canonical)


def extract_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def short_excerpt(text: str, limit: int = 320) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:limit].strip()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def stable_evidence_key(source_url: str, claim: str, quotation: str) -> str:
    digest = sha1(f"{source_url}|{claim}|{quotation}".encode("utf-8")).hexdigest()
    return digest[:16]


def split_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
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


def select_relevant_chunks(
    chunks: Iterable[str],
    *,
    query_terms: Iterable[str],
    max_chunks: int = 4,
) -> list[str]:
    terms = {term.lower() for term in query_terms if term.strip()}
    if not terms:
        return list(chunks)[:max_chunks]
    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        lower = chunk.lower()
        score = sum(1 for term in terms if term in lower)
        scored.append((score, chunk))
    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [chunk for _, chunk in scored[:max_chunks] if chunk.strip()]


def score_candidate(
    candidate: SearchCandidate,
    *,
    active_subqueries: list[Subquery],
    visited_urls: dict[str, object],
    domain_counts: Counter[str],
) -> SearchCandidate:
    score = candidate.score
    reasons: list[str] = []

    if candidate.url in visited_urls:
        score -= 1.0
        reasons.append("already_visited")

    if candidate.snippet:
        score += min(len(candidate.snippet) / 180.0, 1.0)
        reasons.append("has_snippet")

    matched_subqueries = 0
    for subquery in active_subqueries:
        haystack = f"{candidate.title} {candidate.snippet}".lower()
        terms = [token.lower() for token in subquery.search_terms or subquery.question.split()]
        if any(term in haystack for term in terms[:6]):
            matched_subqueries += 1
    if matched_subqueries:
        score += 1.25 * matched_subqueries
        reasons.append("matches_subquery")

    if domain_counts[candidate.domain] == 0:
        score += 0.3
        reasons.append("domain_diversity")

    if candidate.url.startswith("https://"):
        score += 0.1
        reasons.append("https")

    candidate.score = round(score, 4)
    candidate.reasons = reasons
    return candidate


def deduplicate_candidates(
    candidates: Iterable[SearchCandidate],
    *,
    visited_urls: dict[str, object],
) -> tuple[list[SearchCandidate], list[tuple[str, SourceDiscardReason, str]]]:
    unique: dict[str, SearchCandidate] = {}
    discarded: list[tuple[str, SourceDiscardReason, str]] = []
    for candidate in candidates:
        canonical = canonicalize_url(candidate.url)
        candidate.url = canonical
        candidate.domain = candidate.domain or extract_domain(canonical)
        if canonical in visited_urls:
            discarded.append((canonical, SourceDiscardReason.ALREADY_VISITED, "URL ya procesada"))
            continue
        if canonical in unique:
            discarded.append((canonical, SourceDiscardReason.DUPLICATE_URL, "URL duplicada en la misma tanda"))
            continue
        unique[canonical] = candidate
    return list(unique.values()), discarded


def classify_browser_payload(
    *,
    content: str,
    error: str | None,
    exit_code: int | None,
    min_partial_chars: int,
    min_useful_chars: int,
) -> BrowserPageStatus:
    if error:
        lowered = error.lower()
        if any(token in lowered for token in ("404", "not found", "dns", "timed out", "connection refused")):
            return BrowserPageStatus.RECOVERABLE_ERROR
        return BrowserPageStatus.TERMINAL_ERROR

    lowered_content = content.lower()
    if any(token in lowered_content for token in ("captcha", "access denied", "robot check", "cloudflare")):
        return BrowserPageStatus.BLOCKED

    useful_chars = len(content.strip())
    if exit_code not in (None, 0):
        return BrowserPageStatus.RECOVERABLE_ERROR
    if useful_chars >= min_useful_chars:
        return BrowserPageStatus.USEFUL
    if useful_chars >= min_partial_chars:
        return BrowserPageStatus.PARTIAL
    return BrowserPageStatus.EMPTY


def deduplicate_evidence(
    existing: Iterable[AtomicEvidence],
    incoming: Iterable[AtomicEvidence],
) -> list[AtomicEvidence]:
    known = {
        stable_evidence_key(item.source_url, item.claim, item.quotation)
        for item in existing
    }
    accepted: list[AtomicEvidence] = []
    for item in incoming:
        key = stable_evidence_key(item.source_url, item.claim, item.quotation)
        if key in known:
            continue
        known.add(key)
        accepted.append(item)
    return accepted


def update_working_dossier(
    dossier: WorkingDossier,
    *,
    evidence: Iterable[AtomicEvidence],
    source_url: str | None = None,
    source_title: str | None = None,
) -> WorkingDossier:
    merged = dossier.model_copy(deep=True)
    for item in evidence:
        summary_line = f"[{item.subquery_id}] {item.claim}"
        if summary_line not in merged.evidence_digest:
            merged.evidence_digest.append(summary_line)
        current = merged.subquery_summaries.get(item.subquery_id, "")
        combined = "\n".join(part for part in (current, f"- {item.claim}") if part).strip()
        merged.subquery_summaries[item.subquery_id] = combined
        points = merged.subquery_key_points.setdefault(item.subquery_id, [])
        if item.claim not in points:
            points.append(item.claim)
        key_point = f"{item.source_title}: {item.summary}"
        if key_point not in merged.key_points:
            merged.key_points.append(key_point)

    if source_url and source_title:
        source_summary = " | ".join(entry.claim for entry in evidence)
        if source_summary:
            merged.source_summaries[source_url] = f"{source_title}: {source_summary[:400]}"

    merged.global_summary = "\n".join(merged.key_points[:12])[:4000]
    return merged


def compute_minimum_coverage(
    *,
    active_subqueries: list[Subquery],
    evidence: Iterable[AtomicEvidence],
) -> tuple[list[str], list[Gap]]:
    by_subquery = Counter(item.subquery_id for item in evidence)
    resolved_ids: list[str] = []
    gaps: list[Gap] = []
    for subquery in active_subqueries:
        count = by_subquery[subquery.id]
        if count >= subquery.evidence_target:
            resolved_ids.append(subquery.id)
            continue
        missing = subquery.evidence_target - count
        gaps.append(
            Gap(
                subquery_id=subquery.id,
                description=f"Faltan {missing} evidencias para responder con soporte minimo.",
                severity=GapSeverity.HIGH if missing >= 2 else GapSeverity.MEDIUM,
                rationale=subquery.question,
                suggested_queries=subquery.search_terms[:3] or [subquery.question],
                actionable=True,
            )
        )
    return resolved_ids, gaps


def select_evidence_for_context(
    evidence: Iterable[AtomicEvidence],
    *,
    subquery_ids: Iterable[str],
    budget_tokens: int,
) -> list[AtomicEvidence]:
    wanted = set(subquery_ids)
    selected: list[AtomicEvidence] = []
    consumed = 0
    for item in sorted(evidence, key=lambda entry: entry.relevance_score, reverse=True):
        if wanted and item.subquery_id not in wanted:
            continue
        candidate_tokens = estimate_tokens(item.claim + item.summary + item.quotation)
        if consumed + candidate_tokens > budget_tokens:
            break
        selected.append(item)
        consumed += candidate_tokens
    return selected


def build_report_sources(evidence: Iterable[AtomicEvidence]) -> list[ReportSource]:
    grouped: dict[str, dict[str, list[str] | str]] = {}
    for item in evidence:
        bucket = grouped.setdefault(item.source_url, {"title": item.source_title, "evidence_ids": []})
        evidence_ids = bucket["evidence_ids"]
        if item.id not in evidence_ids:
            evidence_ids.append(item.id)
    return [
        ReportSource(url=url, title=str(data["title"]), evidence_ids=list(data["evidence_ids"]))
        for url, data in grouped.items()
    ]


def render_markdown_report(report: FinalReport) -> str:
    lines = [
        "# Informe de investigacion",
        "",
        "## Pregunta de investigacion",
        report.query,
        "",
        "## Resumen ejecutivo",
        report.executive_answer.strip() or "No se pudo redactar una respuesta ejecutiva con soporte suficiente.",
        "",
    ]

    if report.key_findings:
        lines.append("## Hallazgos clave")
        lines.append("")
        for finding in report.key_findings:
            lines.append(f"- {finding}")
        lines.append("")

    for section in report.sections:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.summary.strip():
            lines.append(section.summary.strip())
            lines.append("")
        if section.body.strip():
            lines.append(section.body.strip())
            lines.append("")
        if section.evidence_ids:
            lines.append(f"Evidencia asociada: {', '.join(section.evidence_ids)}")
            lines.append("")

    lines.append("## Confianza y reservas")
    lines.append("")
    lines.append(f"Nivel de confianza: {report.confidence.value}")
    lines.append("")
    if report.reservations:
        for item in report.reservations:
            lines.append(f"- {item}")
    else:
        lines.append("- No se registraron reservas adicionales durante la sintesis final.")
    lines.append("")

    lines.append("## Huecos abiertos")
    lines.append("")
    if report.open_gaps:
        for gap in report.open_gaps:
            lines.append(f"- {gap}")
    else:
        lines.append("- No quedaron huecos abiertos relevantes segun el criterio de cierre aplicado.")
    lines.append("")

    lines.append("## Fuentes citadas")
    lines.append("")
    if report.cited_sources:
        for index, source in enumerate(report.cited_sources, start=1):
            evidence_refs = ", ".join(source.evidence_ids) if source.evidence_ids else "sin evidencia enlazada"
            lines.append(f"{index}. [{source.title}]({source.url}) - evidencia: {evidence_refs}")
    else:
        lines.append("1. No se registraron fuentes citadas.")

    if report.stop_reason:
        lines.extend([
            "",
            "## Cierre de investigacion",
            "",
            f"Motivo de cierre: {report.stop_reason}",
        ])

    return "\n".join(lines).strip() + "\n"
