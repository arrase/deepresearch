"""Bounded and auditable LLM workers."""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple, TypeVar

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
    Subquery,
    coerce_bool
)
from .utils import build_report_sources

TModel = TypeVar("TModel", bound=BaseModel)


class LLMInvocation(NamedTuple):
    content: str
    usage: dict[str, int]


class PlannerPayload(BaseModel):
    subqueries: list[Subquery] = Field(default_factory=list)
    search_intents: list[SearchIntent] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)

    @field_validator("subqueries", mode="before")
    @classmethod
    def validate_subqueries(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return []
        cleaned = []
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
        }
        if json_format:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    def _extract_usage(self, response: Any) -> dict[str, int]:
        usage: dict[str, int] = {}
        candidates = []
        response_metadata = getattr(response, "response_metadata", None)
        usage_metadata = getattr(response, "usage_metadata", None)
        if isinstance(response_metadata, dict):
            candidates.append(response_metadata.get("token_usage"))
            candidates.append(response_metadata.get("usage"))
            candidates.append(response_metadata)
        if isinstance(usage_metadata, dict):
            candidates.append(usage_metadata)

        key_map = {
            "input_tokens": ("input_tokens", "prompt_eval_count", "prompt_tokens"),
            "output_tokens": ("output_tokens", "eval_count", "completion_tokens"),
            "total_tokens": ("total_tokens", "total_token_count"),
        }
        for output_key, input_keys in key_map.items():
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                for input_key in input_keys:
                    value = candidate.get(input_key)
                    if isinstance(value, int):
                        usage[output_key] = value
                        break
                if output_key in usage:
                    break
        if "total_tokens" not in usage and {"input_tokens", "output_tokens"} <= usage.keys():
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        return usage

    def _invoke(self, llm: ChatOllama, messages: list[SystemMessage | HumanMessage]) -> LLMInvocation:
        response = llm.invoke(messages)
        return LLMInvocation(content=str(response.content), usage=self._extract_usage(response))

    def _parse_response(self, prompt_name: str, variables: dict[str, Any], schema: type[TModel], temperature: float) -> tuple[TModel, dict[str, int]]:
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
        
        raw_text = ""
        usage: dict[str, int] = {}
        try:
            invocation = self._invoke(llm, messages)
            raw_text = invocation.content
            usage = invocation.usage
            parsed = self._try_parse(parser, raw_text, schema)
            if parsed:
                return parsed, usage
        except Exception as e:
            raw_text = f"Error: {str(e)}"

        # Repair attempt
        try:
            repair_prompt = self._prompt_loader.render("repair", {
                "format_instructions": instructions,
                "original_prompt": f"SYSTEM: {prompt_pair.system}\nHUMAN: {prompt_pair.human}",
                "raw_output": str(raw_text),
                "parse_error": "Invalid JSON or schema mismatch"
            })
            retry_invocation = self._invoke(llm, [SystemMessage(content=repair_prompt.system), HumanMessage(content=repair_prompt.human)])
            parsed = self._try_parse(parser, retry_invocation.content, schema)
            if parsed:
                return parsed, retry_invocation.usage or usage
        except Exception:
            pass
            
        raise ValueError(f"Failed to parse {prompt_name} response after repair attempt. Raw: {raw_text[:200]}...")

    def _try_parse(self, parser: PydanticOutputParser, text: str, schema: type[TModel]) -> TModel | None:
        try:
            return parser.parse(text)
        except Exception:
            # Salvage JSON block
            match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
            if not match:
                return None
            
            clean_text = match.group(1)
            
            # Attempt 1: Direct load and validate
            try:
                data = json.loads(clean_text)
                return schema.model_validate(data)
            except Exception: pass
            
            # Attempt 2: Clean trailing commas
            try:
                fixed = re.sub(r",\s*([\]\}])", r"\1", clean_text)
                data = json.loads(fixed)
                return schema.model_validate(data)
            except Exception: pass
            
            # Attempt 3: Wrap list in dict if schema expects a wrapped list
            if clean_text.startswith("["):
                try:
                    data = json.loads(clean_text)
                    if isinstance(data, list):
                        # Heuristic: Find first list field in schema
                        for name, field in schema.model_fields.items():
                            if "list" in str(field.annotation).lower():
                                try:
                                    return schema.model_validate({name: data})
                                except: continue
                except: pass

            return None

    def plan_research(self, context: NodeContext) -> PlannerPayload:
        payload, _ = self._parse_response("planner", context.model_dump(), PlannerPayload, self._config.model.temperature_planner)
        return payload

    def plan_research_with_usage(self, context: NodeContext) -> tuple[PlannerPayload, dict[str, int]]:
        return self._parse_response("planner", context.model_dump(), PlannerPayload, self._config.model.temperature_planner)

    def extract_evidence(self, context: NodeContext) -> EvidencePayload:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(f"- {e.claim} | {e.source_title}" for e in context.evidentiary)
        try:
            payload, _ = self._parse_response("extractor", vars, EvidencePayload, self._config.model.temperature_extractor)
            return payload
        except Exception: return EvidencePayload()

    def extract_evidence_with_usage(self, context: NodeContext) -> tuple[EvidencePayload, dict[str, int]]:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(f"- {e.claim} | {e.source_title}" for e in context.evidentiary)
        try:
            return self._parse_response("extractor", vars, EvidencePayload, self._config.model.temperature_extractor)
        except Exception:
            return EvidencePayload(), {}

    def evaluate_coverage(self, context: NodeContext) -> CoveragePayload:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(
            f"- {e.subquery_id}: {e.claim} | source={e.source_title} | domain={re.sub(r'^www\\.', '', e.source_url.split('/')[2])}"
            for e in context.evidentiary
        )
        try:
            payload, _ = self._parse_response("evaluator", vars, CoveragePayload, self._config.model.temperature_evaluator)
            return payload
        except Exception: return CoveragePayload()

    def evaluate_coverage_with_usage(self, context: NodeContext) -> tuple[CoveragePayload, dict[str, int]]:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(
            f"- {e.subquery_id}: {e.claim} | source={e.source_title} | domain={re.sub(r'^www\\.', '', e.source_url.split('/')[2])}"
            for e in context.evidentiary
        )
        try:
            return self._parse_response("evaluator", vars, CoveragePayload, self._config.model.temperature_evaluator)
        except Exception:
            return CoveragePayload(), {}

    def synthesize_report(self, context: NodeContext, query: str) -> FinalReport:
        report, _ = self.synthesize_report_with_usage(context, query)
        return report

    def synthesize_report_with_usage(self, context: NodeContext, query: str) -> tuple[FinalReport, dict[str, int]]:
        vars = context.model_dump()
        vars.update({"query": query, "evidentiary": "\n".join(f"- {e.id}: {e.claim}" for e in context.evidentiary)})
        
        prompt = self._prompt_loader.render("synthesizer", {**vars, "language": self._config.runtime.language, "format_instructions": ""})
        llm = self._llm(temperature=self._config.model.temperature_synthesizer, json_format=False)
        usage: dict[str, int] = {}
        
        try:
            invocation = self._invoke(llm, [SystemMessage(content=prompt.system), HumanMessage(content=prompt.human)])
            raw_text = invocation.content
            usage = invocation.usage
        except Exception as e:
            raw_text = f"# Research Report\n\nError generating report: {str(e)}"
            
        cited_sources = build_report_sources(context.evidentiary)
        # Clean markdown wrappers
        md = raw_text.strip()
        if md.startswith("```"):
            md = re.sub(r"^```(?:markdown)?\n", "", md)
            md = re.sub(r"\n```$", "", md)
        
        if cited_sources:
            md += "\n\n## Referenced Sources\n\n"
            for i, s in enumerate(cited_sources, 1):
                md += f"{i}. [{s.title}]({s.url}) (Evidence: {', '.join(s.evidence_ids)})\n"
                
        return FinalReport(
            query=query, executive_answer="See report.",
            cited_sources=cited_sources, evidence_ids=[e.id for e in context.evidentiary],
            markdown_report=md
        ), usage
