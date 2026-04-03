"""Structured LLM payload models for planner, extractor, and evaluator."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ..state import ConfidenceLevel, Contradiction, Gap, SearchIntent, Subquery, coerce_bool


class PlannerPayload(BaseModel):
    subqueries: list[Subquery] = Field(default_factory=list)
    search_intents: list[SearchIntent] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)

    @field_validator("subqueries", mode="before")
    @classmethod
    def validate_subqueries(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        cleaned: list[Subquery | dict[str, Any]] = []
        for sq in v:
            if isinstance(sq, Subquery):
                cleaned.append(sq)
                continue
            if not isinstance(sq, dict):
                continue

            sq_data = dict(sq)
            # Ensure search_terms is populated
            if not sq_data.get("search_terms"):
                sq_data["search_terms"] = [sq_data.get("question", "")]
            cleaned.append(sq_data)
        return cleaned


class EvidenceDraft(BaseModel):
    summary: str
    claim: str
    quotation: str
    citation_locator: str = "unknown"
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    caveats: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("relevance_score", mode="before")
    @classmethod
    def coerce_float(cls, v: Any) -> float:
        if isinstance(v, str):
            match = re.search(r"\d+(?:\.\d+)?", v)
            return float(match.group(0)) if match else 0.5
        return float(v) if isinstance(v, (int, float)) else 0.5


class EvidencePayload(BaseModel):
    evidences: list[EvidenceDraft] = Field(default_factory=list)


class CoveragePayload(BaseModel):
    resolved_subquery_ids: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    open_gaps: list[Gap] = Field(default_factory=list)
    is_sufficient: bool = False
    rationale: str = ""

    @field_validator("is_sufficient", mode="before")
    @classmethod
    def validate_bool(cls, v: Any) -> bool:
        return coerce_bool(v)
