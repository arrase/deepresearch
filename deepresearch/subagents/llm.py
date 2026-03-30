"""Workers LLM acotados y auditables.

Cada worker usa prompts cortos, contexto minimo y parseo estructurado con
PydanticOutputParser. Cuando el modelo devuelve salida imperfecta se intenta una
reparacion prudente, seguida de un retry controlado y un fallback por extraccion
de JSON si la salida es casi util.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, TypeVar

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

from ..config import ModelConfig, RuntimeConfig
from ..context_manager import NodeContext
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
    def __init__(self, model_config: ModelConfig, runtime_config: RuntimeConfig) -> None:
        self._model_config = model_config
        self._runtime_config = runtime_config

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
        prompt: ChatPromptTemplate,
        variables: dict[str, Any],
        schema: type[TModel],
        temperature: float,
    ) -> TModel:
        parser = PydanticOutputParser(pydantic_object=schema)
        prompt_value = prompt.invoke({**variables, "format_instructions": self._compact_format_instructions(schema)})
        llm = self._llm(temperature=temperature)
        raw = llm.invoke(prompt_value).content
        if isinstance(raw, list):
            raw = "\n".join(str(item) for item in raw)
        raw_text = str(raw)
        parsed = self._try_parse(parser, raw_text, schema)
        if parsed is not None:
            return parsed

        repair_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Repara una salida estructurada invalida sin cambiar el contenido factual. Devuelve solo JSON valido. {format_instructions}",
                ),
                (
                    "human",
                    "Instrucciones originales:\n{original_prompt}\n\nSalida invalida:\n{raw_output}\n\nError de parseo: {parse_error}",
                ),
            ]
        )
        last_error = "initial_parse_failed"
        for _ in range(self._runtime_config.llm_retry_attempts):
            try:
                repair_value = repair_prompt.invoke(
                    {
                        "format_instructions": self._compact_format_instructions(schema),
                        "original_prompt": prompt_value.to_string(),
                        "raw_output": raw_text,
                        "parse_error": last_error,
                    }
                )
                retry_raw = llm.invoke(repair_value).content
                if isinstance(retry_raw, list):
                    retry_raw = "\n".join(str(item) for item in retry_raw)
                retry_text = str(retry_raw)
                parsed = self._try_parse(parser, retry_text, schema)
                if parsed is not None:
                    return parsed
                last_error = "repair_parse_failed"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
        self._dump_failed_output(schema, prompt_value.to_string(), raw_text, last_error)
        raise ValueError(f"No se pudo parsear la salida estructurada: {last_error}")

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

    def _compact_format_instructions(self, schema: type[TModel]) -> str:
        instructions = {
            PlannerPayload: (
                'Devuelve un JSON objeto con las claves: '
                'subqueries=[{id,question,rationale,priority,evidence_target,success_criteria,search_terms}], '
                'search_intents=[{query,rationale,subquery_ids}], hypotheses=[string]. '
                'priority debe ser entero 1-5, evidence_target debe ser entero 1-4, success_criteria debe ser array de strings cortos, '
                'subquery_ids debe contener ids reales de subqueries. '
                'Manten todo muy breve: maximo 3 subqueries, 3 search_intents, 2 success_criteria por subquery y 2 search_terms por subquery. '
                'No anadas texto fuera del JSON.'
            ),
            EvidencePayload: (
                'Devuelve un JSON objeto con la clave evidences=[{summary,claim,quotation,citation_locator,relevance_score,confidence,caveats,tags}]. '
                'Manten la salida breve: maximo 3 evidences y caveats cortos. '
                'confidence debe ser low, medium o high. No anadas texto fuera del JSON.'
            ),
            CoveragePayload: (
                'Devuelve un JSON objeto con las claves resolved_subquery_ids=[string], contradictions=[{topic,statement_a,statement_b,evidence_ids,severity,note}], '
                'open_gaps=[{subquery_id,description,severity,rationale,suggested_queries,actionable}], is_sufficient=bool, rationale=string. '
                'severity debe ser low, medium, high o critical. No anadas texto fuera del JSON.'
            ),
            FinalReportPayload: (
                'Devuelve un JSON objeto con las claves executive_answer=string, key_findings=[string], confidence=low|medium|high, '
                'reservations=[string], open_gaps=[string], '
                'sections=[{title,summary,body,evidence_ids,subquery_ids}]. '
                'Usa maximo 4 sections. Cada section debe citar evidence_ids existentes en la evidencia de entrada. '
                'No anadas texto fuera del JSON.'
            ),
        }
        return instructions.get(schema, 'Devuelve solo un JSON valido sin texto adicional.')

    def _dump_failed_output(self, schema: type[TModel], prompt_text: str, raw_text: str, error: str) -> None:
        target_dir = Path(self._runtime_config.logs_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"llm_parse_failure_{schema.__name__}.json"
        payload = {
            "schema": schema.__name__,
            "error": error,
            "prompt": prompt_text,
            "raw_output": raw_text,
        }
        target.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

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
            raise ValueError("Planner payload no es un objeto JSON")

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
                    "question": str(item.get("question") or "Subpregunta sin texto"),
                    "rationale": str(item.get("rationale") or "Sin razonamiento explicito"),
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
            raise ValueError("Evidence payload no es un objeto JSON")
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
            raise ValueError("Coverage payload no es un objeto JSON")
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
            raise ValueError("Final report payload no es un objeto JSON")
        sections = payload.get("sections", [])
        if isinstance(sections, dict):
            sections = [sections]
        normalized_sections = []
        for item in sections:
            if not isinstance(item, dict):
                continue
            normalized_sections.append(
                {
                    "title": str(item.get("title") or "Analisis"),
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
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Eres un planificador de investigacion. Produce solo agenda, no conclusiones. {format_instructions}",
                ),
                (
                    "human",
                    "{permanent}\n\n{strategic}\n\nTarea: {operational}\nLimita el numero de subpreguntas a 3 y el numero de search_intents a 3. Usa ids estables tipo sq_1, sq_2. evidence_target es un entero pequeno, no una descripcion. rationale debe ser muy corta. success_criteria debe tener como maximo 2 frases cortas. search_terms debe tener como maximo 2 consultas cortas. hypotheses debe tener como maximo 2 frases cortas.",
                ),
            ]
        )
        payload = self._parse_response(
            prompt=prompt,
            variables=context.model_dump(),
            schema=PlannerPayload,
            temperature=self._model_config.temperature_planner,
        )
        for subquery in payload.subqueries:
            if not subquery.search_terms:
                subquery.search_terms = [subquery.question]
        return payload

    def extract_evidence(self, context: NodeContext) -> EvidencePayload:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Extrae evidencia atomica y trazable. No inventes hechos. {format_instructions}",
                ),
                (
                    "human",
                    "{permanent}\n\n{strategic}\n\nEvidencia previa:\n{evidentiary}\n\nFuente actual:\n{local_source}\n\nTarea: {operational}\nMaximo 3 evidencias. caveats y summary deben ser breves.",
                ),
            ]
        )
        variables = context.model_dump()
        variables["evidentiary"] = "\n".join(
            f"- {item.claim} | fuente={item.source_title} | cita={item.citation_locator}"
            for item in context.evidentiary
        )
        return self._parse_response(
            prompt=prompt,
            variables=variables,
            schema=EvidencePayload,
            temperature=self._model_config.temperature_extractor,
        )

    def evaluate_coverage(self, context: NodeContext) -> CoveragePayload:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Evalua cobertura y huecos con prudencia. No cierres la investigacion sin soporte suficiente. {format_instructions}",
                ),
                (
                    "human",
                    "{permanent}\n\n{strategic}\n\nEvidencia:\n{evidentiary}\n\nTarea: {operational}",
                ),
            ]
        )
        variables = context.model_dump()
        variables["evidentiary"] = "\n".join(
            f"- {item.subquery_id}: {item.claim} | confianza={item.confidence.value}"
            for item in context.evidentiary
        )
        payload = self._parse_response(
            prompt=prompt,
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
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Redacta un informe final profundo y estrictamente respaldado por la evidencia proporcionada. No inventes secciones sin soporte. {format_instructions}",
                ),
                (
                    "human",
                    "Pregunta: {query}\n\nContexto estrategico:\n{strategic}\n\nEvidencia:\n{evidentiary}\n\nTarea: {operational}\nRedacta un resumen ejecutivo, hallazgos clave y sections tematicas. Cada section debe apoyarse en evidence_ids existentes.",
                ),
            ]
        )
        variables = context.model_dump()
        variables["query"] = query
        variables["evidentiary"] = "\n".join(
            f"- evidence_id={item.id} | subquery_id={item.subquery_id} | claim={item.claim} | fuente={item.source_title} | cita={item.citation_locator} | confianza={item.confidence.value} | caveats={'; '.join(item.caveats[:2])}"
            for item in context.evidentiary
        )
        payload = self._parse_response(
            prompt=prompt,
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
                    title="Analisis principal",
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
