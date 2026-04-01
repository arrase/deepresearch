"""Structured state and evidence models for the research system.

The state is not a bag of text. Every field exists to support graph decisions
and maintain traceability between the query, evidence, visited sources, and the
final report.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class BrowserPageStatus(str, Enum):
    USEFUL = "useful"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    EMPTY = "empty"
    RECOVERABLE_ERROR = "recoverable_error"
    TERMINAL_ERROR = "terminal_error"


class SourceDiscardReason(str, Enum):
    DUPLICATE_URL = "duplicate_url"
    ALREADY_VISITED = "already_visited"
    BLOCKED = "blocked"
    EMPTY = "empty"
    LOW_VALUE = "low_value"
    TECHNICAL_ERROR = "technical_error"
    NO_EVIDENCE = "no_evidence"
    IRRELEVANT = "irrelevant"


class GapSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SearchIntent(BaseModel):
    query: str
    rationale: str
    subquery_ids: list[str] = Field(default_factory=list)


import re

def coerce_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.lower() in {"true", "yes", "1", "si", "sí", "true.", "yes."}
    return bool(v)

def coerce_int(v: Any, default: int = 1) -> int:
    if isinstance(v, int): return v
    try:
        match = re.search(r"\d+", str(v))
        return int(match.group()) if match else default
    except: return default

class Subquery(BaseModel):
    id: str = Field(default_factory=lambda: f"sq_{uuid4().hex[:10]}")
    question: str
    rationale: str
    status: Literal["active", "resolved", "discarded"] = "active"
    priority: int = 1
    evidence_target: int = 2
    success_criteria: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)

    @field_validator("priority", mode="before")
    @classmethod
    def clamp_priority(cls, v: Any) -> int:
        val = coerce_int(v, 1)
        return max(1, min(5, val))

    @field_validator("evidence_target", mode="before")
    @classmethod
    def clamp_evidence(cls, v: Any) -> int:
        val = coerce_int(v, 2)
        return max(1, min(10, val))


class SearchCandidate(BaseModel):
    url: str
    title: str
    snippet: str = ""
    domain: str = ""
    source_type: str = "web"
    score: float = 0.0
    reasons: list[str] = Field(default_factory=list)
    subquery_ids: list[str] = Field(default_factory=list)
    discovered_via: str = "search"


class DiscardedSource(BaseModel):
    url: str
    reason: SourceDiscardReason
    note: str = ""
    timestamp: str = Field(default_factory=utc_now_iso)


class SourceVisit(BaseModel):
    url: str
    final_url: str | None = None
    title: str = ""
    status: BrowserPageStatus
    content: str = ""
    excerpt: str = ""
    error: str | None = None
    candidate_subquery_ids: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    fetched_at: str = Field(default_factory=utc_now_iso)


class AtomicEvidence(BaseModel):
    id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    subquery_id: str
    source_url: str
    source_title: str
    summary: str
    claim: str
    quotation: str
    citation_locator: str
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    caveats: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    extracted_at: str = Field(default_factory=utc_now_iso)


class Contradiction(BaseModel):
    id: str = Field(default_factory=lambda: f"cx_{uuid4().hex[:10]}")
    topic: str
    statement_a: str
    statement_b: str
    evidence_ids: list[str] = Field(default_factory=list)
    severity: GapSeverity = GapSeverity.MEDIUM
    note: str = ""


class Gap(BaseModel):
    id: str = Field(default_factory=lambda: f"gap_{uuid4().hex[:10]}")
    subquery_id: str
    description: str
    severity: GapSeverity = GapSeverity.MEDIUM
    rationale: str = ""
    suggested_queries: list[str] = Field(default_factory=list)
    actionable: bool = True


class WorkingDossier(BaseModel):
    subquery_summaries: dict[str, str] = Field(default_factory=dict)
    key_points: list[str] = Field(default_factory=list)
    source_summaries: dict[str, str] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now_iso)


class ReportSource(BaseModel):
    url: str
    title: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    title: str
    summary: str
    body: str
    evidence_ids: list[str] = Field(default_factory=list)
    subquery_ids: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    query: str
    executive_answer: str
    key_findings: list[str] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    reservations: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)
    cited_sources: list[ReportSource] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    markdown_report: str = ""
    markdown_artifact_path: str | None = None
    stop_reason: str | None = None
    context_window_tokens: int | None = None
    reserved_output_tokens: int | None = None
    prompt_tokens: int | None = None
    evidence_tokens: int | None = None
    available_prompt_tokens: int | None = None
    llm_usage: dict[str, int] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=utc_now_iso)


class TelemetryEvent(BaseModel):
    timestamp: str = Field(default_factory=utc_now_iso)
    stage: str
    message: str
    verbosity: int = Field(default=1, ge=0, le=3)
    payload_type: str = "generic"
    payload: dict[str, Any] = Field(default_factory=dict)


class ResearchState(TypedDict):
    query: str
    active_subqueries: list[Subquery]
    resolved_subqueries: list[Subquery]
    search_intents: list[SearchIntent]
    completed_search_queries: list[str]
    search_queue: list[SearchCandidate]
    visited_urls: dict[str, SourceVisit]
    discarded_sources: list[DiscardedSource]
    atomic_evidence: list[AtomicEvidence]
    contradictions: list[Contradiction]
    open_gaps: list[Gap]
    working_dossier: WorkingDossier
    final_report: FinalReport | None
    is_sufficient: bool
    hypotheses: list[str]
    iteration: int
    max_iterations: int
    stagnation_cycles: int
    consecutive_technical_failures: int
    cycles_without_new_evidence: int
    cycles_without_useful_sources: int
    progress_score: int
    useful_sources_count: int
    urls_visited_since_eval: int
    telemetry: list[TelemetryEvent]
    stop_reason: str | None
    technical_reason: str | None
    llm_usage: NotRequired[dict[str, dict[str, int]]]
    synthesis_budget: NotRequired[dict[str, int | bool | str | None]]
    current_candidate: NotRequired[SearchCandidate | None]
    current_browser_result: NotRequired[SourceVisit | None]
    latest_evidence: NotRequired[list[AtomicEvidence]]


def build_initial_state(
    query: str,
    *,
    max_iterations: int,
) -> ResearchState:
    return {
        "query": query,
        "active_subqueries": [],
        "resolved_subqueries": [],
        "search_intents": [],
        "completed_search_queries": [],
        "search_queue": [],
        "visited_urls": {},
        "discarded_sources": [],
        "atomic_evidence": [],
        "contradictions": [],
        "open_gaps": [],
        "working_dossier": WorkingDossier(),
        "final_report": None,
        "is_sufficient": False,
        "hypotheses": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "stagnation_cycles": 0,
        "consecutive_technical_failures": 0,
        "cycles_without_new_evidence": 0,
        "cycles_without_useful_sources": 0,
        "progress_score": 0,
        "useful_sources_count": 0,
        "urls_visited_since_eval": 0,
        "telemetry": [],
        "stop_reason": None,
        "technical_reason": None,
        "llm_usage": {},
        "synthesis_budget": {},
        "current_candidate": None,
        "current_browser_result": None,
        "latest_evidence": [],
    }
