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

from pydantic import BaseModel, ConfigDict, Field


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
    model_config = ConfigDict(extra="forbid")

    query: str
    rationale: str
    subquery_ids: list[str] = Field(default_factory=list)


class Subquery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"sq_{uuid4().hex[:10]}")
    question: str
    rationale: str
    status: Literal["active", "resolved", "discarded"] = "active"
    priority: int = Field(default=1, ge=1, le=5)
    evidence_target: int = Field(default=2, ge=1, le=10)
    success_criteria: list[str] = Field(default_factory=list)
    search_terms: list[str] = Field(default_factory=list)


class SearchCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    url: str
    reason: SourceDiscardReason
    note: str = ""
    timestamp: str = Field(default_factory=utc_now_iso)


class BrowserResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    status: BrowserPageStatus
    title: str = ""
    content: str = ""
    excerpt: str = ""
    final_url: str | None = None
    exit_code: int | None = None
    error: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    fetched_at: str = Field(default_factory=utc_now_iso)


class SourceVisit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    final_url: str | None = None
    title: str = ""
    status: BrowserPageStatus
    content_excerpt: str = ""
    error: str | None = None
    candidate_subquery_ids: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    fetched_at: str = Field(default_factory=utc_now_iso)


class AtomicEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"cx_{uuid4().hex[:10]}")
    topic: str
    statement_a: str
    statement_b: str
    evidence_ids: list[str] = Field(default_factory=list)
    severity: GapSeverity = GapSeverity.MEDIUM
    note: str = ""


class Gap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: f"gap_{uuid4().hex[:10]}")
    subquery_id: str
    description: str
    severity: GapSeverity = GapSeverity.MEDIUM
    rationale: str = ""
    suggested_queries: list[str] = Field(default_factory=list)
    actionable: bool = True


class ContextWindowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_tokens: int
    configured_by: str
    selection_policy: str


class WorkingDossier(BaseModel):
    model_config = ConfigDict(extra="forbid")

    global_summary: str = ""
    subquery_summaries: dict[str, str] = Field(default_factory=dict)
    subquery_key_points: dict[str, list[str]] = Field(default_factory=dict)
    source_summaries: dict[str, str] = Field(default_factory=dict)
    key_points: list[str] = Field(default_factory=list)
    evidence_digest: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now_iso)


class ReportSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str
    title: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    summary: str
    body: str
    evidence_ids: list[str] = Field(default_factory=list)
    subquery_ids: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    generated_at: str = Field(default_factory=utc_now_iso)


class TelemetryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: str = Field(default_factory=utc_now_iso)
    stage: str
    message: str
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
    context_window_config: ContextWindowConfig
    final_report: FinalReport | None
    is_sufficient: bool
    hypotheses: list[str]
    iteration: int
    max_iterations: int
    telemetry: list[TelemetryEvent]
    fallback_reason: str | None
    current_candidate: NotRequired[SearchCandidate | None]
    current_browser_result: NotRequired[BrowserResult | None]
    latest_evidence: NotRequired[list[AtomicEvidence]]


def build_initial_state(
    query: str,
    *,
    max_iterations: int,
    target_tokens: int,
    configured_by: str,
    selection_policy: str,
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
        "context_window_config": ContextWindowConfig(
            target_tokens=target_tokens,
            configured_by=configured_by,
            selection_policy=selection_policy,
        ),
        "final_report": None,
        "is_sufficient": False,
        "hypotheses": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "telemetry": [],
        "fallback_reason": None,
        "current_candidate": None,
        "current_browser_result": None,
        "latest_evidence": [],
    }
