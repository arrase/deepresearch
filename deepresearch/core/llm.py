"""Bounded and auditable LLM workers."""

from __future__ import annotations

import json
import re
from typing import Any, Literal, NamedTuple, TypeVar, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from langsmith import traceable
from pydantic import BaseModel, ValidationError

from ..config import ResearchConfig
from ..context_manager import NodeContext
from ..prompting import PromptTemplateLoader
from ..state import FinalReport
from .payloads import CoveragePayload, EvidencePayload, PlannerPayload
from .utils import build_report_sources

_JSON_FALLBACK_INSTRUCTION = "Return valid JSON only."

TModel = TypeVar("TModel", bound=BaseModel)


class LLMInvocation(NamedTuple):
    content: str
    usage: dict[str, int]


class LLMWorkers:
    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._prompt_loader = PromptTemplateLoader(config.prompts_dir, strict_templates=True)

    def _llm(self, temperature: float, json_format: bool = True) -> ChatOllama:
        format_value: Literal["json"] | None = "json" if json_format else None
        return ChatOllama(
            model=self._config.model.model_name,
            base_url=self._config.model.base_url,
            temperature=temperature,
            num_ctx=self._config.model.num_ctx,
            num_predict=self._config.model.num_predict,
            disable_streaming=True,
            format=format_value,
            sync_client_kwargs={"timeout": self._config.model.timeout_seconds},
        )

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

    def _invoke(self, llm: ChatOllama, messages: list[BaseMessage]) -> LLMInvocation:
        response = llm.invoke(messages)
        return LLMInvocation(content=str(response.content), usage=self._extract_usage(response))

    def _parse_response(
        self,
        prompt_name: str,
        variables: dict[str, Any],
        schema: type[TModel],
        temperature: float,
    ) -> tuple[TModel, dict[str, int]]:
        parser = PydanticOutputParser(pydantic_object=schema)
        try:
            instructions = self._prompt_loader.render_format(prompt_name, variables)
        except (FileNotFoundError, OSError, ValueError):
            instructions = _JSON_FALLBACK_INSTRUCTION

        prompt_pair = self._prompt_loader.render(
            prompt_name,
            {
                **variables,
                "language": self._config.runtime.language,
                "format_instructions": instructions,
            },
        )

        llm = self._llm(temperature=temperature)
        messages = [SystemMessage(content=prompt_pair.system), HumanMessage(content=prompt_pair.human)]

        raw_text = ""
        usage: dict[str, int] = {}
        parse_error = "Invalid JSON or schema mismatch"
        try:
            invocation = self._invoke(llm, messages)
            raw_text = invocation.content
            usage = invocation.usage
            parsed = self._try_parse(parser, raw_text, schema)
            if parsed:
                return parsed, usage
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
            parse_error = str(e)
            raw_text = f"Error: {e}"

        # Repair attempt
        repair_raw_text = ""
        repair_usage: dict[str, int] = {}
        try:
            repair_prompt = self._prompt_loader.render(
                "repair",
                {
                    "format_instructions": instructions,
                    "original_prompt": f"SYSTEM: {prompt_pair.system}\nHUMAN: {prompt_pair.human}",
                    "raw_output": str(raw_text),
                    "parse_error": parse_error,
                },
            )
            retry_invocation = self._invoke(
                llm,
                [
                    SystemMessage(content=repair_prompt.system),
                    HumanMessage(content=repair_prompt.human),
                ],
            )
            repair_raw_text = retry_invocation.content
            repair_usage = retry_invocation.usage or usage
            parsed = self._try_parse(parser, repair_raw_text, schema)
            if parsed:
                return parsed, repair_usage
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError, OSError):
            pass

        raise ValueError(f"Failed to parse {prompt_name} response after repair attempt. Raw: {raw_text[:200]}...")

    def _try_parse(self, parser: PydanticOutputParser, text: str, schema: type[TModel]) -> TModel | None:
        try:
            return cast(TModel, parser.parse(text))
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

    @traceable(name="planner-llm")
    def plan_research(self, context: NodeContext) -> PlannerPayload:
        payload, _ = self._parse_response(
            "planner",
            context.model_dump(),
            PlannerPayload,
            self._config.model.temperature_planner,
        )
        return payload

    @traceable(name="planner-llm-with-usage")
    def plan_research_with_usage(self, context: NodeContext) -> tuple[PlannerPayload, dict[str, int]]:
        return self._parse_response(
            "planner",
            context.model_dump(),
            PlannerPayload,
            self._config.model.temperature_planner,
        )

    @traceable(name="extractor-llm")
    def extract_evidence(self, context: NodeContext) -> EvidencePayload:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(f"- {e.claim} | {e.source_title}" for e in context.evidentiary)
        try:
            payload, _ = self._parse_response(
                "extractor",
                vars,
                EvidencePayload,
                self._config.model.temperature_extractor,
            )
            return payload
        except (json.JSONDecodeError, ValidationError, ValueError):
            return EvidencePayload()

    @traceable(name="extractor-llm-with-usage")
    def extract_evidence_with_usage(self, context: NodeContext) -> tuple[EvidencePayload, dict[str, int]]:
        vars = context.model_dump()
        vars["evidentiary"] = "\n".join(f"- {e.claim} | {e.source_title}" for e in context.evidentiary)
        try:
            return self._parse_response(
                "extractor",
                vars,
                EvidencePayload,
                self._config.model.temperature_extractor,
            )
        except (json.JSONDecodeError, ValidationError, ValueError):
            return EvidencePayload(), {}

    @staticmethod
    def _extract_domain_label(url: str) -> str:
        return re.sub(r"^www\.", "", url.split("/")[2])

    @traceable(name="evaluator-llm")
    def evaluate_coverage(self, context: NodeContext) -> CoveragePayload:
        vars = context.model_dump()
        _domain = self._extract_domain_label
        vars["evidentiary"] = "\n".join(
            f"- {e.subquery_id}: {e.claim} | source={e.source_title} | domain={_domain(e.source_url)}"
            for e in context.evidentiary
        )
        try:
            payload, _ = self._parse_response(
                "evaluator",
                vars,
                CoveragePayload,
                self._config.model.temperature_evaluator,
            )
            return payload
        except (json.JSONDecodeError, ValidationError, ValueError):
            return CoveragePayload()

    @traceable(name="evaluator-llm-with-usage")
    def evaluate_coverage_with_usage(self, context: NodeContext) -> tuple[CoveragePayload, dict[str, int]]:
        vars = context.model_dump()
        _domain = self._extract_domain_label
        vars["evidentiary"] = "\n".join(
            f"- {e.subquery_id}: {e.claim} | source={e.source_title} | domain={_domain(e.source_url)}"
            for e in context.evidentiary
        )
        try:
            return self._parse_response(
                "evaluator",
                vars,
                CoveragePayload,
                self._config.model.temperature_evaluator,
            )
        except (json.JSONDecodeError, ValidationError, ValueError):
            return CoveragePayload(), {}

    def synthesize_report(self, context: NodeContext, query: str) -> FinalReport:
        report, _ = self.synthesize_report_with_usage(context, query)
        return report

    @traceable(name="synthesizer-llm-with-usage")
    def synthesize_report_with_usage(self, context: NodeContext, query: str) -> tuple[FinalReport, dict[str, int]]:
        vars = context.model_dump()
        vars.update({
            "query": query,
            "evidentiary": "\n".join(f"- {e.id}: {e.claim}" for e in context.evidentiary),
        })

        prompt = self._prompt_loader.render(
            "synthesizer",
            {
                **vars,
                "language": self._config.runtime.language,
                "format_instructions": "",
            },
        )
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
