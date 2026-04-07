"""Canonical state and domain models for the research pipeline.

The hierarchical Map-Reduce architecture uses chapters (depth=0) that
decompose into sub-topics (depth>=1).  Each chapter is researched
independently, synthesised into a ``ChapterDraft``, and then all drafts
are assembled into the ``FinalReport`` by a global synthesiser.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {
            "true", "yes", "1", "si", "si.", "sí", "sí.", "true.", "yes.",
        }
    return bool(value)


def coerce_int(value: Any, default: int = 1) -> int:
    if isinstance(value, int):
        return value
    match = re.search(r"\d+", str(value))
    return int(match.group(0)) if match else default


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Search and planning models
# ---------------------------------------------------------------------------

class SearchIntent(BaseModel):
    query: str
    rationale: str
    topic_ids: list[str] = Field(default_factory=list)


class ResearchTopic(BaseModel):
    """A research topic in the hierarchical plan tree.

    *  ``depth == 0`` -> chapter (meta-planner output).
    *  ``depth >= 1`` -> sub-topic created by the micro-planner or auditor.
    *  ``chapter_id`` always points to the root chapter it belongs to.
    """

    id: str = Field(default_factory=lambda: f"topic_{uuid4().hex[:10]}")
    question: str
    rationale: str
    success_criteria: list[str] = Field(default_factory=list)
    status: TopicStatus = TopicStatus.PENDING
    priority: int = 1
    evidence_target: int = 1
    search_terms: list[str] = Field(default_factory=list)
    last_query: str | None = None

    # -- hierarchical fields --
    parent_id: str | None = None
    depth: int = 0
    chapter_id: str = ""

    @field_validator("priority", mode="before")
    @classmethod
    def clamp_priority(cls, value: Any) -> int:
        return max(1, min(5, coerce_int(value, 1)))

    @field_validator("evidence_target", mode="before")
    @classmethod
    def clamp_evidence_target(cls, value: Any) -> int:
        return max(1, coerce_int(value, 1))

    @field_validator("depth", mode="before")
    @classmethod
    def clamp_depth(cls, value: Any) -> int:
        return max(0, coerce_int(value, 0))


# ---------------------------------------------------------------------------
# Search history models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evidence models
# ---------------------------------------------------------------------------

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
    chapter_id: str = ""
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


# ---------------------------------------------------------------------------
# Synthesis budget
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Chapter draft (sub-synthesiser output — lives in RAM only)
# ---------------------------------------------------------------------------

class ChapterDraft(BaseModel):
    """Structured intermediate output produced by the sub-synthesiser.

    One ``ChapterDraft`` is generated per completed chapter.  The global
    synthesiser later consumes all drafts to produce the final report.
    """

    chapter_id: str
    title: str
    executive_summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    key_findings: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    cited_sources: list[ReportSource] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    limitations: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Final report
# ---------------------------------------------------------------------------

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
    stop_reason: str | None = None
    context_window_tokens: int | None = None
    reserved_output_tokens: int | None = None
    prompt_tokens: int | None = None
    evidence_tokens: int | None = None
    available_prompt_tokens: int | None = None
    llm_usage: dict[str, int] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=utc_now_iso)


# ---------------------------------------------------------------------------
# Master research state (TypedDict for LangGraph)
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    # -- query & iteration --
    query: str
    max_iterations: int
    current_iteration: int

    # -- hierarchical plan --
    plan: list[ResearchTopic]
    active_topic_id: str | None
    topic_attempts: dict[str, int]
    current_chapter_id: str | None
    completed_chapter_ids: list[str]
    flushed_chapter_ids: list[str]

    # -- chapter drafts (Map-Reduce) --
    chapter_drafts: list[ChapterDraft]
    topic_audit_attempts: dict[str, int]

    # -- search --
    search_intents: list[SearchIntent]
    hypotheses: list[str]
    search_history: list[SearchAttempt]
    completed_search_queries: list[str]
    failed_queries: list[str]
    candidate_queue: list[SearchCandidate]
    current_batch: list[SearchCandidate]

    # -- sources --
    visited_urls: dict[str, SourceRecord]
    discarded_sources: list[DiscardedSource]

    # -- evidence pipeline --
    extracted_evidence_buffer: list[EvidenceDraft]
    curated_evidence: list[CuratedEvidence]
    accumulated_evidence_tokens_exact: int
    accumulated_evidence_tokens_prompt_fit: int

    # -- evaluation & coverage --
    synthesis_budget: SynthesisBudget
    topic_coverage: dict[str, TopicCoverage]
    open_gaps: list[Gap]
    contradictions: list[Contradiction]

    # -- stagnation counters --
    cycles_without_new_evidence: int
    cycles_without_useful_sources: int
    consecutive_empty_search_cycles: int
    consecutive_technical_failures: int
    new_evidence_in_cycle: int
    merged_evidence_in_cycle: int
    useful_source_in_cycle: bool

    # -- control flow --
    stop_reason: str | None
    stop_details: str | None
    technical_reason: str | None
    audit_approved: bool

    # -- working memory --
    working_dossier: WorkingDossier
    llm_usage: dict[str, dict[str, int]]
    final_report: FinalReport | None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_initial_state(query: str, *, max_iterations: int) -> ResearchState:
    return {
        "query": query,
        "max_iterations": max_iterations,
        "current_iteration": 0,
        # plan
        "plan": [],
        "active_topic_id": None,
        "topic_attempts": {},
        "current_chapter_id": None,
        "completed_chapter_ids": [],
        "flushed_chapter_ids": [],
        # chapter drafts
        "chapter_drafts": [],
        "topic_audit_attempts": {},
        # search
        "search_intents": [],
        "hypotheses": [],
        "search_history": [],
        "completed_search_queries": [],
        "failed_queries": [],
        "candidate_queue": [],
        "current_batch": [],
        # sources
        "visited_urls": {},
        "discarded_sources": [],
        # evidence
        "extracted_evidence_buffer": [],
        "curated_evidence": [],
        "accumulated_evidence_tokens_exact": 0,
        "accumulated_evidence_tokens_prompt_fit": 0,
        # evaluation
        "synthesis_budget": SynthesisBudget(),
        "topic_coverage": {},
        "open_gaps": [],
        "contradictions": [],
        # counters
        "cycles_without_new_evidence": 0,
        "cycles_without_useful_sources": 0,
        "consecutive_empty_search_cycles": 0,
        "consecutive_technical_failures": 0,
        "new_evidence_in_cycle": 0,
        "merged_evidence_in_cycle": 0,
        "useful_source_in_cycle": False,
        # control
        "stop_reason": None,
        "stop_details": None,
        "technical_reason": None,
        "audit_approved": False,
        # working memory
        "working_dossier": WorkingDossier(),
        "llm_usage": {},
        "final_report": None,
    }
