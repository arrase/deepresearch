"""Bounded and auditable LLM workers."""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ValidationError

from ..config import ResearchConfig
from ..context_manager import NodeContext
from ..prompting import PromptTemplateLoader
from ..state import FinalReport, TelemetryEvent
from ..telemetry import TelemetryRecorder
from .payloads import CoveragePayload, EvidencePayload, PlannerPayload
from .utils import build_report_sources, estimate_tokens

_JSON_FALLBACK_INSTRUCTION = "Return valid JSON only."

TModel = TypeVar("TModel", bound=BaseModel)


class LLMInvocation(NamedTuple):
    content: str
    usage: dict[str, int]


class LLMWorkers:
    def __init__(self, config: ResearchConfig, telemetry: TelemetryRecorder | None = None) -> None:
        self._config = config
        self._telemetry = telemetry
        self._pending_telemetry: list[TelemetryEvent] = []
        self._prompt_loader = PromptTemplateLoader(config.prompts_dir, strict_templates=True)

    def consume_telemetry_events(self) -> list[TelemetryEvent]:
        pending = self._pending_telemetry
        self._pending_telemetry = []
        return pending

    def _record_debug_event(
        self,
        stage: str,
        message: str,
        *,
        verbosity: int = 2,
        payload_type: str = "llm_response",
        **payload: Any,
    ) -> None:
        if self._telemetry is None:
            return
        event = self._telemetry.record(
            stage,
            message,
            verbosity=verbosity,
            payload_type=payload_type,
            **payload,
        )
        if event is not None:
            self._pending_telemetry.append(event)

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
        except (FileNotFoundError, OSError, ValueError):
            instructions = _JSON_FALLBACK_INSTRUCTION

        prompt_pair = self._prompt_loader.render(prompt_name, {
            **variables,
            "language": self._config.runtime.language,
            "format_instructions": instructions
        })

        llm = self._llm(temperature=temperature)
        messages = [SystemMessage(content=prompt_pair.system), HumanMessage(content=prompt_pair.human)]
        prompt_tokens = estimate_tokens(prompt_pair.system) + estimate_tokens(prompt_pair.human)

        raw_text = ""
        usage: dict[str, int] = {}
        parse_error = "Invalid JSON or schema mismatch"
        try:
            invocation = self._invoke(llm, messages)
            raw_text = invocation.content
            usage = invocation.usage
            parsed = self._try_parse(parser, raw_text, schema)
            if parsed:
                self._record_debug_event(
                    prompt_name,
                    "LLM response parsed",
                    worker=prompt_name,
                    temperature=temperature,
                    prompt_tokens=prompt_tokens,
                    usage=usage,
                    raw_output=raw_text,
                    parsed_output=parsed.model_dump(mode="json"),
                    repair_attempted=False,
                )
                return parsed, usage
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
            parse_error = str(e)
            raw_text = f"Error: {e}"

        self._record_debug_event(
            prompt_name,
            "LLM response requires repair",
            payload_type="llm_repair",
            worker=prompt_name,
            temperature=temperature,
            prompt_tokens=prompt_tokens,
            usage=usage,
            raw_output=raw_text,
            parse_error=parse_error,
        )

        # Repair attempt
        repair_error = None
        repair_raw_text = ""
        repair_usage: dict[str, int] = {}
        try:
            repair_prompt = self._prompt_loader.render("repair", {
                "format_instructions": instructions,
                "original_prompt": f"SYSTEM: {prompt_pair.system}\nHUMAN: {prompt_pair.human}",
                "raw_output": str(raw_text),
                "parse_error": parse_error
            })
            retry_invocation = self._invoke(llm, [SystemMessage(content=repair_prompt.system), HumanMessage(content=repair_prompt.human)])
            repair_raw_text = retry_invocation.content
            repair_usage = retry_invocation.usage or usage
            parsed = self._try_parse(parser, repair_raw_text, schema)
            if parsed:
                self._record_debug_event(
                    prompt_name,
                    "LLM repair parsed",
                    worker=prompt_name,
                    temperature=temperature,
                    prompt_tokens=prompt_tokens,
                    usage=repair_usage,
                    raw_output=repair_raw_text,
                    parsed_output=parsed.model_dump(mode="json"),
                    repair_attempted=True,
                    parse_error=parse_error,
                )
                return parsed, repair_usage
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError, OSError) as exc:
            repair_error = str(exc)

        self._record_debug_event(
            prompt_name,
            "LLM repair failed",
            payload_type="llm_repair",
            worker=prompt_name,
            temperature=temperature,
            prompt_tokens=prompt_tokens,
            usage=repair_usage or usage,
            raw_output=repair_raw_text or raw_text,
            parse_error=parse_error,
            repair_error=repair_error,
        )

        raise ValueError(f"Failed to parse {prompt_name} response after repair attempt. Raw: {raw_text[:200]}...")

    def _try_parse(self, parser: PydanticOutputParser, text: str, schema: type[TModel]) -> TModel | None:
        try:
            return parser.parse(text)
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError):
            pass

        # Salvage JSON block from surrounding text
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        if not match:
            return None

        clean_text = match.group(1)

        # Attempt 1: Direct load and validate
        try:
            data = json.loads(clean_text)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Attempt 2: Clean trailing commas
        try:
            fixed = re.sub(r",\s*([\]\}])", r"\1", clean_text)
            data = json.loads(fixed)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Attempt 3: Wrap list in dict if schema expects a wrapped list
        if clean_text.startswith("["):
            try:
                data = json.loads(clean_text)
                if isinstance(data, list):
                    for name, field_info in schema.model_fields.items():
                        if "list" in str(field_info.annotation).lower():
                            try:
                                return schema.model_validate({name: data})
                            except (ValidationError, ValueError):
                                continue
            except (json.JSONDecodeError, ValidationError):
                pass

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
        except (json.JSONDecodeError, ValidationError, ValueError):
            return EvidencePayload()

    def extract_evidence_with_usage(self, context: NodeContext) -> tuple[EvidencePayload, dict[str, int]]:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(f"- {e.claim} | {e.source_title}" for e in context.evidentiary)
        try:
            return self._parse_response("extractor", vars, EvidencePayload, self._config.model.temperature_extractor)
        except (json.JSONDecodeError, ValidationError, ValueError):
            return EvidencePayload(), {}

    @staticmethod
    def _extract_domain_label(url: str) -> str:
        return re.sub(r"^www\.", "", url.split("/")[2])

    def evaluate_coverage(self, context: NodeContext) -> CoveragePayload:
        vars = context.model_dump()
        _domain = self._extract_domain_label
        vars["evidentiary"] = "\n".join(
            f"- {e.subquery_id}: {e.claim} | source={e.source_title} | domain={_domain(e.source_url)}"
            for e in context.evidentiary
        )
        try:
            payload, _ = self._parse_response("evaluator", vars, CoveragePayload, self._config.model.temperature_evaluator)
            return payload
        except (json.JSONDecodeError, ValidationError, ValueError):
            return CoveragePayload()

    def evaluate_coverage_with_usage(self, context: NodeContext) -> tuple[CoveragePayload, dict[str, int]]:
        vars = context.model_dump()
        _domain = self._extract_domain_label
        vars["evidentiary"] = "\n".join(
            f"- {e.subquery_id}: {e.claim} | source={e.source_title} | domain={_domain(e.source_url)}"
            for e in context.evidentiary
        )
        try:
            return self._parse_response("evaluator", vars, CoveragePayload, self._config.model.temperature_evaluator)
        except (json.JSONDecodeError, ValidationError, ValueError):
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
        except (ValueError, KeyError, OSError) as e:
            raw_text = f"# Research Report\n\nError generating report: {e}"

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
