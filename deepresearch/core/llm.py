"""Bounded and auditable LLM workers."""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, field_validator

from ..config import ResearchConfig
from ..context_manager import NodeContext
from ..prompting import PromptTemplateLoader
from ..state import (
    ConfidenceLevel, 
    Contradiction, 
    FinalReport, 
    Gap, 
    SearchIntent, 
    Subquery
)
from .utils import build_report_sources

TModel = TypeVar("TModel", bound=BaseModel)


class PlannerPayload(BaseModel):
    subqueries: list[Subquery] = Field(default_factory=list)
    search_intents: list[SearchIntent] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)

    @field_validator("subqueries", mode="before")
    @classmethod
    def validate_subqueries(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        # Ensure search_terms is populated if missing
        cleaned = []
        for sq in v:
            if isinstance(sq, Subquery):
                cleaned.append(sq)
                continue
            if not isinstance(sq, dict):
                continue
            
            # Use a copy to avoid mutating input if it's a dict
            sq_data = dict(sq)
            if not sq_data.get("search_terms"):
                sq_data["search_terms"] = [sq_data.get("question", "")]
            # Small models often output priority/evidence_target as strings
            if isinstance(sq_data.get("priority"), str):
                try: sq_data["priority"] = int(re.search(r"\d+", sq_data["priority"]).group())
                except: sq_data["priority"] = 1
            if isinstance(sq_data.get("evidence_target"), str):
                try: sq_data["evidence_target"] = int(re.search(r"\d+", sq_data["evidence_target"]).group())
                except: sq_data["evidence_target"] = 3
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
    def coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() in {"true", "yes", "1", "si", "sí"}
        return bool(v)


class FinalReportPayload(BaseModel):
    class SectionPayload(BaseModel):
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
        self._prompt_loader = PromptTemplateLoader(config.prompts_dir, strict_templates=True)

    def _llm(self, temperature: float, json_format: bool = True) -> ChatOllama:
        kwargs = {
            "model": self._config.model.model_name,
            "base_url": self._config.model.base_url,
            "temperature": temperature,
            "num_ctx": self._config.model.num_ctx,
            "num_predict": self._config.model.num_predict,
            "timeout": self._config.model.timeout_seconds,
            "disable_streaming": True,
            "reasoning": False,
        }
        if json_format:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    def _parse_response(self, prompt_name: str, variables: dict[str, Any], schema: type[TModel], temperature: float) -> TModel:
        parser = PydanticOutputParser(pydantic_object=schema)
        try:
            instructions = self._prompt_loader.render_format(prompt_name, variables)
        except Exception:
            instructions = "Return valid JSON only."
            
        prompt_pair = self._prompt_loader.render(prompt_name, {
            **variables, 
            "language": self._config.runtime.language, 
            "format_instructions": instructions
        })
        
        llm = self._llm(temperature=temperature)
        messages = [SystemMessage(content=prompt_pair.system), HumanMessage(content=prompt_pair.human)]
        
        try:
            raw_text = llm.invoke(messages).content
            if isinstance(raw_text, list): raw_text = "\n".join(map(str, raw_text))
            
            parsed = self._try_parse(parser, str(raw_text), schema)
            if parsed: return parsed
        except Exception as e:
            raw_text = f"Error during LLM invocation: {str(e)}"

        # Simple retry with repair
        try:
            repair_prompt = self._prompt_loader.render("repair", {
                "format_instructions": instructions,
                "original_prompt": f"SYSTEM: {prompt_pair.system}\nHUMAN: {prompt_pair.human}",
                "raw_output": str(raw_text),
                "parse_error": "JSON Parse/Validation Error"
            })
            retry_raw = llm.invoke([SystemMessage(content=repair_prompt.system), HumanMessage(content=repair_prompt.human)]).content
            parsed = self._try_parse(parser, str(retry_raw), schema)
            if parsed: return parsed
        except Exception:
            pass
            
        raise ValueError(f"Failed to parse {prompt_name} response after repair attempt")

    def _try_parse(self, parser: PydanticOutputParser, text: str, schema: type[TModel]) -> TModel | None:
        try:
            return parser.parse(text)
        except Exception:
            # Basic salvage: extract JSON block
            match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if not match:
                return None
            
            clean_text = match.group(1)
            
            # Attempt 1: Direct load
            try:
                return schema.model_validate(json.loads(clean_text))
            except Exception: pass
            
            # Attempt 2: Clean common syntax errors (trailing commas, etc)
            try:
                # Remove trailing commas before closing braces/brackets
                fixed = re.sub(r",\s*([\]\}])", r"\1", clean_text)
                return schema.model_validate(json.loads(fixed))
            except Exception: pass
            
            # Attempt 3: If it starts with [, wrap it in a dict if the schema expects one
            # (though Pydantic models usually expect a dict at root)
            if clean_text.startswith("[") and not issubclass(schema, list):
                # This is a bit speculative, but some models return a list of objects 
                # instead of a wrapped object.
                try:
                    data = json.loads(clean_text)
                    if isinstance(data, list):
                        # Try to guess the field name or just try to validate it
                        # For PlannerPayload, it expects 'subqueries', 'search_intents', etc.
                        # This part is complex to generalize, so we'll keep it simple.
                        pass
                except Exception: pass

            return None

    def plan_research(self, context: NodeContext) -> PlannerPayload:
        return self._parse_response("planner", context.model_dump(), PlannerPayload, self._config.model.temperature_planner)

    def extract_evidence(self, context: NodeContext) -> EvidencePayload:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(f"- {e.claim} | {e.source_title}" for e in context.evidentiary)
        try:
            return self._parse_response("extractor", vars, EvidencePayload, self._config.model.temperature_extractor)
        except Exception: return EvidencePayload()

    def evaluate_coverage(self, context: NodeContext) -> CoveragePayload:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(f"- {e.subquery_id}: {e.claim}" for e in context.evidentiary)
        try:
            return self._parse_response("evaluator", vars, CoveragePayload, self._config.model.temperature_evaluator)
        except Exception: return CoveragePayload()

    def synthesize_report(self, context: NodeContext, query: str) -> FinalReport:
        vars = context.model_dump()
        vars.update({"query": query, "evidentiary": "\n".join(f"- {e.id}: {e.claim}" for e in context.evidentiary)})
        
        prompt = self._prompt_loader.render("synthesizer", {**vars, "language": self._config.runtime.language, "format_instructions": ""})
        llm = self._llm(temperature=self._config.model.temperature_synthesizer, json_format=False)
        raw_text = str(llm.invoke([SystemMessage(content=prompt.system), HumanMessage(content=prompt.human)]).content)
        
        cited_sources = build_report_sources(context.evidentiary)
        md = raw_text.strip().strip("`").replace("markdown", "", 1).strip()
        
        if cited_sources:
            md += "\n\n## Referenced Sources\n\n"
            for i, s in enumerate(cited_sources, 1):
                md += f"{i}. [{s.title}]({s.url}) (Evidence: {', '.join(s.evidence_ids)})\n"
                
        return FinalReport(
            query=query, executive_answer="See report.",
            cited_sources=cited_sources, evidence_ids=[e.id for e in context.evidentiary],
            markdown_report=md
        )
