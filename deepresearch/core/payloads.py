"""Structured LLM payload models for all pipeline stages."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..state import ConfidenceLevel, ResearchTopic, SearchIntent, TopicStatus, coerce_bool

# ---------------------------------------------------------------------------
# Shared validator helpers
# ---------------------------------------------------------------------------

def _normalize_topic_list(value: Any) -> list[ResearchTopic | dict[str, Any]]:
    """Coerce a raw list into validated topic dicts/objects."""
    if not isinstance(value, list):
        return []
    cleaned: list[ResearchTopic | dict[str, Any]] = []
    for topic in value:
        if isinstance(topic, ResearchTopic):
            cleaned.append(topic)
            continue
        if not isinstance(topic, dict):
            continue
        topic_data = dict(topic)
        if not topic_data.get("search_terms"):
            topic_data["search_terms"] = [topic_data.get("question", "")]
        status = str(topic_data.get("status", TopicStatus.PENDING.value)).strip().lower()
        if status == "active":
            topic_data["status"] = TopicStatus.PENDING.value
        elif status == "resolved":
            topic_data["status"] = TopicStatus.COMPLETED.value
        elif status == "discarded":
            topic_data["status"] = TopicStatus.EXHAUSTED.value
        cleaned.append(topic_data)
    return cleaned


# ---------------------------------------------------------------------------
# Meta-planner (query -> chapters)
# ---------------------------------------------------------------------------

class MetaPlannerPayload(BaseModel):
    """Output of the meta-planner: high-level chapter breakdown."""

    chapters: list[ResearchTopic] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)

    @field_validator("chapters", mode="before")
    @classmethod
    def validate_chapters(cls, value: Any) -> Any:
        return _normalize_topic_list(value)


# ---------------------------------------------------------------------------
# Micro-planner (chapter -> sub-topics)
# ---------------------------------------------------------------------------

class MicroPlannerPayload(BaseModel):
    """Output of the micro-planner: sub-topics for one chapter."""

    subtopics: list[ResearchTopic] = Field(default_factory=list)
    search_intents: list[SearchIntent] = Field(default_factory=list)

    @field_validator("subtopics", mode="before")
    @classmethod
    def validate_subtopics(cls, value: Any) -> Any:
        return _normalize_topic_list(value)


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------

class EvidenceDraft(BaseModel):
    summary: str
    claim: str
    quotation: str
    citation_locator: str = "unknown"
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    caveats: list[str] = Field(default_factory=list)

    @field_validator("relevance_score", mode="before")
    @classmethod
    def coerce_float(cls, value: Any) -> float:
        if isinstance(value, str):
            match = re.search(r"\d+(?:\.\d+)?", value)
            return float(match.group(0)) if match else 0.5
        return float(value) if isinstance(value, (int, float)) else 0.5


class EvidencePayload(BaseModel):
    evidences: list[EvidenceDraft] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Auditor (devil's advocate)
# ---------------------------------------------------------------------------

class AuditPayload(BaseModel):
    """Output of the auditor node: approve or reject a chapter's evidence."""

    approved: bool = False
    objections: list[str] = Field(default_factory=list)
    suggested_topics: list[ResearchTopic] = Field(default_factory=list)
    unresolved_limitations: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("approved", mode="before")
    @classmethod
    def validate_approved(cls, value: Any) -> bool:
        return coerce_bool(value)

    @field_validator("suggested_topics", mode="before")
    @classmethod
    def validate_suggested(cls, value: Any) -> Any:
        return _normalize_topic_list(value)
