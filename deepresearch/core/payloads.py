"""Structured LLM payload models for planner, extractor, and evaluator."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..state import ConfidenceLevel, Contradiction, Gap, ResearchTopic, SearchIntent, TopicStatus, coerce_bool


class PlannerPayload(BaseModel):
    subqueries: list[ResearchTopic] = Field(default_factory=list)
    search_intents: list[SearchIntent] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)

    @field_validator("subqueries", mode="before")
    @classmethod
    def validate_topics(cls, value: Any) -> Any:
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


class CoveragePayload(BaseModel):
    resolved_subquery_ids: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    open_gaps: list[Gap] = Field(default_factory=list)
    is_sufficient: bool = False
    rationale: str = ""

    @field_validator("contradictions", mode="before")
    @classmethod
    def normalize_contradictions(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []
        normalized: list[Contradiction | dict[str, Any]] = []
        for item in value:
            if isinstance(item, Contradiction):
                normalized.append(item)
                continue
            if not isinstance(item, dict):
                continue
            contradiction_data = dict(item)
            if "topic" in contradiction_data and "topic_id" not in contradiction_data:
                contradiction_data["topic_id"] = contradiction_data.pop("topic")
            normalized.append(contradiction_data)
        return normalized

    @field_validator("open_gaps", mode="before")
    @classmethod
    def normalize_gaps(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return []
        normalized: list[Gap | dict[str, Any]] = []
        for item in value:
            if isinstance(item, Gap):
                normalized.append(item)
                continue
            if not isinstance(item, dict):
                continue
            gap_data = dict(item)
            if "subquery_id" in gap_data and "topic_id" not in gap_data:
                gap_data["topic_id"] = gap_data.pop("subquery_id")
            normalized.append(gap_data)
        return normalized

    @field_validator("is_sufficient", mode="before")
    @classmethod
    def validate_bool(cls, value: Any) -> bool:
        return coerce_bool(value)
