"""Canonical state and domain models for the SLM-oriented research pipeline."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "si", "si.", "sí", "sí.", "true.", "yes."}
    return bool(value)


def coerce_int(value: Any, default: int = 1) -> int:
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else default


class TopicStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"


class StopReason(StrEnum):
    CONTEXT_SATURATION = "context_saturation"
    PLAN_COMPLETED = "plan_completed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    STUCK_NO_SOURCES = "stuck_no_sources"


class SourceDiscardReason(StrEnum):
    DUPLICATE_URL = "duplicate_url"
    ALREADY_VISITED = "already_visited"
    BLOCKED = "blocked"
    EMPTY = "empty"
    LOW_VALUE = "low_value"
    TECHNICAL_ERROR = "technical_error"
    NO_EVIDENCE = "no_evidence"
    IRRELEVANT = "irrelevant"


class GapSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SearchIntent(BaseModel):
    query: str
    rationale: str
    topic_ids: list[str] = Field(default_factory=list)


class ResearchTopic(BaseModel):
    id: str = Field(default_factory=lambda: f"topic_{uuid4().hex[:10]}")
    question: str
    rationale: str
    success_criteria: list[str] = Field(default_factory=list)
    status: TopicStatus = TopicStatus.PENDING
    priority: int = 1
    evidence_target: int = 1
    search_terms: list[str] = Field(default_factory=list)
    last_query: str | None = None

    @field_validator("priority", mode="before")
    @classmethod
    def clamp_priority(cls, value: Any) -> int:
        return max(1, min(5, coerce_int(value, 1)))

    @field_validator("evidence_target", mode="before")
    @classmethod
    def clamp_evidence_target(cls, value: Any) -> int:
        return max(1, coerce_int(value, 1))


class SearchAttempt(BaseModel):
    topic_id: str
    query: str
    iteration: int
    discovered_urls: int
    accepted_urls: int
    repeated_urls: int
    empty_result: bool
    technical_error: str | None = None
    discovered_at: str = Field(default_factory=utc_now_iso)


class SearchCandidate(BaseModel):
    url: str
    normalized_url: str = ""
    title: str = ""
    snippet: str = ""
    domain: str = ""
    score: float = 0.0
    topic_ids: list[str] = Field(default_factory=list)
    discovered_via: str = "search"
    query: str = ""
    raw_content: str = ""


class DiscardedSource(BaseModel):
    url: str
    reason: SourceDiscardReason
    note: str = ""
    timestamp: str = Field(default_factory=utc_now_iso)


class SourceRecord(BaseModel):
    url: str
    final_url: str | None = None
    title: str = ""
    extracted_chars: int = 0
    relevant_chunks: list[str] = Field(default_factory=list)
    topic_ids: list[str] = Field(default_factory=list)
    last_error: str | None = None
    processed_at: str = Field(default_factory=utc_now_iso)


class EvidenceSourceRef(BaseModel):
    url: str
    title: str
    locator: str = "unknown"


class EvidenceDraft(BaseModel):
    id: str = Field(default_factory=lambda: f"draft_{uuid4().hex[:12]}")
    topic_id: str
    source_url: str
    source_title: str
    claim: str
    quotation: str
    locator: str = "unknown"
    summary: str
    extractor_output_tokens: int = 0
    extractor_input_tokens: int = 0
    extraction_confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    caveats: list[str] = Field(default_factory=list)


class CuratedEvidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"evidence_{uuid4().hex[:12]}")
    topic_id: str
    canonical_claim: str
    summary: str
    support_quotes: list[str] = Field(default_factory=list)
    sources: list[EvidenceSourceRef] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    novelty_score: float = Field(default=1.0, ge=0.0, le=1.0)
    exact_generation_tokens: int = 0
    prompt_fit_tokens_estimate: int = 0
    first_seen_iteration: int = 0
    last_confirmed_iteration: int = 0
    merged_from_drafts: list[str] = Field(default_factory=list)
    canonical_fingerprint: str = ""


class Contradiction(BaseModel):
    id: str = Field(default_factory=lambda: f"cx_{uuid4().hex[:10]}")
    topic_id: str
    statement_a: str
    statement_b: str
    evidence_ids: list[str] = Field(default_factory=list)
    severity: GapSeverity = GapSeverity.MEDIUM
    note: str = ""


class Gap(BaseModel):
    id: str = Field(default_factory=lambda: f"gap_{uuid4().hex[:10]}")
    topic_id: str
    description: str
    severity: GapSeverity = GapSeverity.MEDIUM
    rationale: str = ""
    suggested_queries: list[str] = Field(default_factory=list)
    actionable: bool = True


class TopicCoverage(BaseModel):
    topic_id: str
    accepted_evidence_count: int = 0
    unique_domains: int = 0
    attempts: int = 0
    empty_attempts: int = 0
    resolved: bool = False
    exhausted: bool = False
    rationale: str = ""
    pending_gaps: list[str] = Field(default_factory=list)


class WorkingDossier(BaseModel):
    topic_summaries: dict[str, str] = Field(default_factory=dict)
    key_points: list[str] = Field(default_factory=list)
    source_summaries: dict[str, str] = Field(default_factory=dict)
    updated_at: str = Field(default_factory=utc_now_iso)


class SynthesisBudget(BaseModel):
    context_window_tokens: int = 0
    reserved_output_tokens: int = 0
    prompt_margin_tokens: int = 0
    base_prompt_tokens: int = 0
    available_prompt_tokens: int = 0
    selected_evidence_tokens: int = 0
    candidate_evidence_tokens: int = 0
    overflow_tokens: int = 0
    selected_evidence_count: int = 0
    candidate_evidence_count: int = 0
    final_context_full: bool = False


class ReportSource(BaseModel):
    url: str
    title: str
    evidence_ids: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    title: str
    summary: str
    body: str
    evidence_ids: list[str] = Field(default_factory=list)
    topic_ids: list[str] = Field(default_factory=list)


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


class ResearchState(TypedDict):
    query: str
    max_iterations: int
    current_iteration: int
    plan: list[ResearchTopic]
    active_topic_id: str | None
    topic_attempts: dict[str, int]
    search_intents: list[SearchIntent]
    hypotheses: list[str]
    search_history: list[SearchAttempt]
    completed_search_queries: list[str]
    failed_queries: list[str]
    candidate_queue: list[SearchCandidate]
    current_batch: list[SearchCandidate]
    visited_urls: dict[str, SourceRecord]
    discarded_sources: list[DiscardedSource]
    extracted_evidence_buffer: list[EvidenceDraft]
    curated_evidence: list[CuratedEvidence]
    accumulated_evidence_tokens_exact: int
    accumulated_evidence_tokens_prompt_fit: int
    synthesis_budget: SynthesisBudget
    topic_coverage: dict[str, TopicCoverage]
    open_gaps: list[Gap]
    contradictions: list[Contradiction]
    cycles_without_new_evidence: int
    cycles_without_useful_sources: int
    consecutive_empty_search_cycles: int
    consecutive_technical_failures: int
    new_evidence_in_cycle: int
    merged_evidence_in_cycle: int
    useful_source_in_cycle: bool
    stop_reason: str | None
    stop_details: str | None
    technical_reason: str | None
    replan_requested: bool
    working_dossier: WorkingDossier
    llm_usage: dict[str, dict[str, int]]
    final_report: FinalReport | None


def build_initial_state(query: str, *, max_iterations: int) -> ResearchState:
    return {
        "query": query,
        "max_iterations": max_iterations,
        "current_iteration": 0,
        "plan": [],
        "active_topic_id": None,
        "topic_attempts": {},
        "search_intents": [],
        "hypotheses": [],
        "search_history": [],
        "completed_search_queries": [],
        "failed_queries": [],
        "candidate_queue": [],
        "current_batch": [],
        "visited_urls": {},
        "discarded_sources": [],
        "extracted_evidence_buffer": [],
        "curated_evidence": [],
        "accumulated_evidence_tokens_exact": 0,
        "accumulated_evidence_tokens_prompt_fit": 0,
        "synthesis_budget": SynthesisBudget(),
        "topic_coverage": {},
        "open_gaps": [],
        "contradictions": [],
        "cycles_without_new_evidence": 0,
        "cycles_without_useful_sources": 0,
        "consecutive_empty_search_cycles": 0,
        "consecutive_technical_failures": 0,
        "new_evidence_in_cycle": 0,
        "merged_evidence_in_cycle": 0,
        "useful_source_in_cycle": False,
        "stop_reason": None,
        "stop_details": None,
        "technical_reason": None,
        "replan_requested": False,
        "working_dossier": WorkingDossier(),
        "llm_usage": {},
        "final_report": None,
    }
