"""Bounded and auditable LLM workers.

Each worker uses short prompts, minimal context, and structured parsing through
PydanticOutputParser. When the model emits imperfect output, the runtime tries
prudent repair, followed by controlled retries and JSON salvage when the answer
is still close to usable.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

from ..config import ResearchConfig
from ..context_manager import NodeContext
from ..prompting import PromptMessages, PromptTemplateLoader
from ..state import ConfidenceLevel, Contradiction, FinalReport, Gap, GapSeverity, SearchIntent, Subquery
from .deterministic import build_report_sources, render_markdown_report

TModel = TypeVar("TModel", bound=BaseModel)


class PlannerPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subqueries: list[Subquery]
    search_intents: list[SearchIntent]
    hypotheses: list[str] = Field(default_factory=list)


class EvidenceDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    claim: str
    quotation: str
    citation_locator: str
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    caveats: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class EvidencePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidences: list[EvidenceDraft] = Field(default_factory=list)


class CoveragePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolved_subquery_ids: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    open_gaps: list[Gap] = Field(default_factory=list)
    is_sufficient: bool = False
    rationale: str = ""


class FinalReportPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    class SectionPayload(BaseModel):
        model_config = ConfigDict(extra="forbid")

        title: str
        summary: str
        body: str
        evidence_ids: list[str] = Field(default_factory=list)
        subquery_ids: list[str] = Field(default_factory=list)

    executive_answer: str
    key_findings: list[str] = Field(default_factory=list)
    sections: list[SectionPayload] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    reservations: list[str] = Field(default_factory=list)
    open_gaps: list[str] = Field(default_factory=list)


class LLMWorkers:
    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._model_config = config.model
        self._runtime_config = config.runtime
        self._prompt_loader = PromptTemplateLoader(
            config.prompts_dir,
            strict_templates=config.prompts.strict_templates,
        )

    def _llm(self, *, temperature: float) -> ChatOllama:
        return ChatOllama(
            model=self._model_config.model_name,
            base_url=self._model_config.base_url,
            disable_streaming=True,
            reasoning=False,
            temperature=temperature,
            num_ctx=self._model_config.num_ctx,
            num_predict=self._model_config.num_predict,
            timeout=self._model_config.timeout_seconds,
            format="json",
        )

    def _parse_response(
        self,
        *,
        prompt_name: str,
        variables: dict[str, Any],
        schema: type[TModel],
        temperature: float,
    ) -> TModel:
        parser = PydanticOutputParser(pydantic_object=schema)
        format_instructions = self._load_format_instructions(prompt_name, variables)
        prompt_pair = self._render_prompt_pair(
            prompt_name,
            {**variables, "format_instructions": format_instructions},
        )
        prompt_text = self._stringify_prompt(prompt_pair)
        llm = self._llm(temperature=temperature)
        raw = llm.invoke(self._as_langchain_messages(prompt_pair)).content
        if isinstance(raw, list):
            raw = "\n".join(str(item) for item in raw)
        raw_text = str(raw)
        parsed = self._try_parse(parser, raw_text, schema)
        if parsed is not None:
            return parsed

        last_error = "initial_parse_failed"
        for _ in range(self._runtime_config.llm_retry_attempts):
            try:
                repair_pair = self._render_prompt_pair(
                    "repair",
                    {
                        "format_instructions": format_instructions,
                        "original_prompt": prompt_text,
                        "raw_output": raw_text,
                        "parse_error": last_error,
                    }
                )
                retry_raw = llm.invoke(self._as_langchain_messages(repair_pair)).content
                if isinstance(retry_raw, list):
                    retry_raw = "\n".join(str(item) for item in retry_raw)
                retry_text = str(retry_raw)
                parsed = self._try_parse(parser, retry_text, schema)
                if parsed is not None:
                    return parsed
                last_error = "repair_parse_failed"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        raise ValueError(f"Failed to parse structured output: {last_error}")

    def _render_prompt_pair(self, prompt_name: str, variables: dict[str, Any]) -> PromptMessages:
        return self._prompt_loader.render(prompt_name, variables)

    def _as_langchain_messages(self, prompt_pair: PromptMessages) -> list[SystemMessage | HumanMessage]:
        return [
            SystemMessage(content=prompt_pair.system),
            HumanMessage(content=prompt_pair.human),
        ]

    def _stringify_prompt(self, prompt_pair: PromptMessages) -> str:
        return f"SYSTEM:\n{prompt_pair.system}\n\nHUMAN:\n{prompt_pair.human}"

    def _try_parse(
        self,
        parser: PydanticOutputParser,
        raw_text: str,
        schema: type[TModel],
    ) -> TModel | None:
        try:
            return parser.parse(raw_text)
        except Exception:  # noqa: BLE001
            repaired = _extract_json_block(raw_text)
            if repaired is None:
                loaded = self._salvage_payload(schema, raw_text)
                if loaded is None:
                    return None
            else:
                try:
                    loaded = json.loads(repaired)
                except Exception:  # noqa: BLE001
                    loaded = self._salvage_payload(schema, raw_text)
                    if loaded is None:
                        return None
            try:
                normalized = self._normalize_payload(schema, loaded)
                return schema.model_validate(normalized)
            except Exception:  # noqa: BLE001
                return None

    def _load_format_instructions(self, prompt_name: str, variables: dict[str, Any]) -> str:
        try:
            return self._prompt_loader.render_format(prompt_name, variables)
        except Exception:  # noqa: BLE001
            return "Return valid JSON only, with no additional text."

    def _normalize_payload(self, schema: type[TModel], payload: Any) -> Any:
        if schema is PlannerPayload:
            return self._normalize_planner_payload(payload)
        if schema is EvidencePayload:
            return self._normalize_evidence_payload(payload)
        if schema is CoveragePayload:
            return self._normalize_coverage_payload(payload)
        if schema is FinalReportPayload:
            return self._normalize_final_report_payload(payload)
        return payload

    def _salvage_payload(self, schema: type[TModel], raw_text: str) -> Any | None:
        cleaned = _strip_code_fences(raw_text)
        if schema is EvidencePayload:
            return _salvage_evidence_payload(cleaned)
        return None

    def _normalize_planner_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Planner payload is not a JSON object")

        subqueries = payload.get("subqueries", [])
        if isinstance(subqueries, dict):
            subqueries = [subqueries]
        normalized_subqueries: list[dict[str, Any]] = []
        generated_ids: list[str] = []
        for index, item in enumerate(subqueries, start=1):
            if not isinstance(item, dict):
                continue
            subquery_id = str(item.get("id") or f"sq_plan_{index}")
            generated_ids.append(subquery_id)
            normalized_subqueries.append(
                {
                    "id": subquery_id,
                    "question": str(item.get("question") or "Subquery text missing"),
                    "rationale": str(item.get("rationale") or "No explicit rationale provided"),
                    "priority": _coerce_int(item.get("priority"), default=min(index, 5), minimum=1, maximum=5),
                    "evidence_target": _coerce_int(item.get("evidence_target"), default=2, minimum=1, maximum=4),
                    "success_criteria": _ensure_list(item.get("success_criteria")),
                    "search_terms": _ensure_list(item.get("search_terms")) or [str(item.get("question") or "")],
                }
            )

        search_intents = payload.get("search_intents", [])
        if isinstance(search_intents, dict):
            search_intents = [search_intents]
        normalized_intents: list[dict[str, Any]] = []
        for item in search_intents:
            if not isinstance(item, dict):
                continue
            mapped_ids: list[str] = []
            for raw_id in _ensure_list(item.get("subquery_ids")):
                if isinstance(raw_id, int):
                    idx = raw_id - 1
                    if 0 <= idx < len(generated_ids):
                        mapped_ids.append(generated_ids[idx])
                    continue
                raw_text = str(raw_id)
                if raw_text.isdigit():
                    idx = int(raw_text) - 1
                    if 0 <= idx < len(generated_ids):
                        mapped_ids.append(generated_ids[idx])
                    continue
                mapped_ids.append(raw_text)
            normalized_intents.append(
                {
                    "query": str(item.get("query") or ""),
                    "rationale": str(item.get("rationale") or ""),
                    "subquery_ids": mapped_ids,
                }
            )

        hypotheses = _ensure_list(payload.get("hypotheses"))
        return {
            "subqueries": normalized_subqueries,
            "search_intents": normalized_intents,
            "hypotheses": hypotheses,
        }

    def _normalize_evidence_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Evidence payload is not a JSON object")
        evidences = payload.get("evidences", [])
        if isinstance(evidences, dict):
            evidences = [evidences]
        normalized = []
        for item in evidences:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "summary": str(item.get("summary") or item.get("claim") or ""),
                    "claim": str(item.get("claim") or item.get("summary") or ""),
                    "quotation": str(item.get("quotation") or item.get("claim") or ""),
                    "citation_locator": str(item.get("citation_locator") or "unknown"),
                    "relevance_score": _coerce_float(item.get("relevance_score"), default=0.5),
                    "confidence": _coerce_confidence(item.get("confidence")),
                    "caveats": _ensure_list(item.get("caveats")),
                    "tags": _ensure_list(item.get("tags")),
                }
            )
        return {"evidences": normalized}

    def _normalize_coverage_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Coverage payload is not a JSON object")
        contradictions = payload.get("contradictions", [])
        if isinstance(contradictions, dict):
            contradictions = [contradictions]
        open_gaps = payload.get("open_gaps", [])
        if isinstance(open_gaps, dict):
            open_gaps = [open_gaps]
        normalized_contradictions = []
        for item in contradictions:
            if not isinstance(item, dict):
                continue
            normalized_contradictions.append(
                {
                    "topic": str(item.get("topic") or "unknown"),
                    "statement_a": str(item.get("statement_a") or ""),
                    "statement_b": str(item.get("statement_b") or ""),
                    "evidence_ids": _ensure_list(item.get("evidence_ids")),
                    "severity": _coerce_severity(item.get("severity")),
                    "note": str(item.get("note") or ""),
                }
            )
        normalized_gaps = []
        for item in open_gaps:
            if not isinstance(item, dict):
                continue
            normalized_gaps.append(
                {
                    "subquery_id": str(item.get("subquery_id") or "unknown"),
                    "description": str(item.get("description") or ""),
                    "severity": _coerce_severity(item.get("severity")),
                    "rationale": str(item.get("rationale") or ""),
                    "suggested_queries": _ensure_list(item.get("suggested_queries")),
                    "actionable": _coerce_bool(item.get("actionable"), default=True),
                }
            )
        return {
            "resolved_subquery_ids": [str(item) for item in _ensure_list(payload.get("resolved_subquery_ids"))],
            "contradictions": normalized_contradictions,
            "open_gaps": normalized_gaps,
            "is_sufficient": _coerce_bool(payload.get("is_sufficient"), default=False),
            "rationale": str(payload.get("rationale") or ""),
        }

    def _normalize_final_report_payload(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Final report payload is not a JSON object")
        sections = payload.get("sections", [])
        if isinstance(sections, dict):
            sections = [sections]
        normalized_sections = []
        for item in sections:
            if not isinstance(item, dict):
                continue
            normalized_sections.append(
                {
                    "title": str(item.get("title") or "Analysis"),
                    "summary": str(item.get("summary") or ""),
                    "body": str(item.get("body") or item.get("summary") or ""),
                    "evidence_ids": [str(value) for value in _ensure_list(item.get("evidence_ids"))],
                    "subquery_ids": [str(value) for value in _ensure_list(item.get("subquery_ids"))],
                }
            )
        return {
            "executive_answer": str(payload.get("executive_answer") or payload.get("answer") or ""),
            "key_findings": _ensure_list(payload.get("key_findings")),
            "sections": normalized_sections,
            "confidence": _coerce_confidence(payload.get("confidence")),
            "reservations": _ensure_list(payload.get("reservations")),
            "open_gaps": _ensure_list(payload.get("open_gaps")),
        }

    def plan_research(self, context: NodeContext) -> PlannerPayload:
        payload = self._parse_response(
            prompt_name="planner",
            variables=context.model_dump(),
            schema=PlannerPayload,
            temperature=self._model_config.temperature_planner,
        )
        for subquery in payload.subqueries:
            if not subquery.search_terms:
                subquery.search_terms = [subquery.question]
        return payload

    def extract_evidence(self, context: NodeContext) -> EvidencePayload:
        variables = context.model_dump()
        variables["evidentiary"] = "\n".join(
            f"- {item.claim} | source={item.source_title} | citation={item.citation_locator}"
            for item in context.evidentiary
        )
        return self._parse_response(
            prompt_name="extractor",
            variables=variables,
            schema=EvidencePayload,
            temperature=self._model_config.temperature_extractor,
        )

    def evaluate_coverage(self, context: NodeContext) -> CoveragePayload:
        variables = context.model_dump()
        variables["evidentiary"] = "\n".join(
            f"- {item.subquery_id}: {item.claim} | confidence={item.confidence.value}"
            for item in context.evidentiary
        )
        payload = self._parse_response(
            prompt_name="evaluator",
            variables=variables,
            schema=CoveragePayload,
            temperature=self._model_config.temperature_evaluator,
        )
        for gap in payload.open_gaps:
            if not gap.suggested_queries:
                gap.suggested_queries = [gap.description]
            if gap.severity not in {GapSeverity.LOW, GapSeverity.MEDIUM, GapSeverity.HIGH, GapSeverity.CRITICAL}:
                gap.severity = GapSeverity.MEDIUM
        return payload

    def synthesize_report(self, context: NodeContext, *, query: str) -> FinalReport:
        variables = context.model_dump()
        variables["query"] = query
        variables["evidentiary"] = "\n".join(
            f"- evidence_id={item.id} | subquery_id={item.subquery_id} | claim={item.claim} | source={item.source_title} | citation={item.citation_locator} | confidence={item.confidence.value} | caveats={'; '.join(item.caveats[:2])}"
            for item in context.evidentiary
        )
        payload = self._parse_response(
            prompt_name="synthesizer",
            variables=variables,
            schema=FinalReportPayload,
            temperature=self._model_config.temperature_synthesizer,
        )
        evidence_ids = [item.id for item in context.evidentiary]
        report = FinalReport(
            query=query,
            executive_answer=payload.executive_answer,
            key_findings=payload.key_findings,
            sections=payload.sections,
            confidence=payload.confidence,
            reservations=payload.reservations,
            open_gaps=payload.open_gaps,
            evidence_ids=evidence_ids,
            cited_sources=build_report_sources(context.evidentiary),
        )
        if not report.sections and context.evidentiary:
            report.sections = [
                FinalReportPayload.SectionPayload(
                    title="Primary analysis",
                    summary=report.executive_answer,
                    body="\n".join(f"- {item.claim}" for item in context.evidentiary[:4]),
                    evidence_ids=[item.id for item in context.evidentiary[:4]],
                    subquery_ids=[item.subquery_id for item in context.evidentiary[:4]],
                )
            ]
        report.markdown_report = render_markdown_report(report)
        return report


def _extract_json_block(text: str) -> str | None:
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if match is None:
        return None
    candidate = match.group(1)
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return candidate


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_\-]*\n", "", stripped)
        stripped = stripped.removesuffix("```").strip()
    return stripped


def _salvage_evidence_payload(text: str) -> dict[str, Any] | None:
    cleaned = _strip_code_fences(text)
    key_index = cleaned.find('"evidences"')
    if key_index == -1:
        return None
    array_start = cleaned.find("[", key_index)
    if array_start == -1:
        return None
    objects: list[dict[str, Any]] = []
    depth = 0
    current: list[str] = []
    in_string = False
    escape = False
    collecting = False
    for ch in cleaned[array_start + 1 :]:
        if collecting:
            current.append(ch)
        if ch == '"' and not escape:
            in_string = not in_string
        if ch == "\\" and not escape:
            escape = True
            continue
        escape = False
        if in_string:
            continue
        if ch == "{" and not collecting:
            collecting = True
            current = [ch]
            depth = 1
            continue
        if collecting and ch == "{":
            depth += 1
        elif collecting and ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    objects.append(json.loads("".join(current)))
                except Exception:  # noqa: BLE001
                    pass
                collecting = False
                current = []
        elif ch == "]" and not collecting:
            break
    if not objects:
        return None
    return {"evidences": objects}


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(minimum, min(maximum, value))
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match:
            return max(minimum, min(maximum, int(match.group(0))))
    return default


def _coerce_float(value: Any, *, default: float) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            return max(0.0, min(1.0, float(match.group(0))))
    return default


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "si", "sí"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return default


def _coerce_confidence(value: Any) -> str:
    lowered = str(value or "medium").strip().lower()
    if lowered in {"low", "medium", "high"}:
        return lowered
    return "medium"


def _coerce_severity(value: Any) -> str:
    lowered = str(value or "medium").strip().lower()
    if lowered in {"low", "medium", "high", "critical"}:
        return lowered
    return "medium"
