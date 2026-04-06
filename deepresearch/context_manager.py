"""Prompt context assembly and synthesis budget calculation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

from pydantic import BaseModel, Field

from .config import ResearchConfig
from .core.utils import estimate_tokens, extract_domain, select_evidence_for_context, summarize_gaps
from .prompting import PromptTemplateLoader
from .state import CuratedEvidence, Gap, ResearchState, ResearchTopic, SynthesisBudget, TopicBrief, TopicStatus


class NodeContext(BaseModel):
    query: str
    has_subqueries: bool = False
    coverage_summary: str = ""
    source_balance_summary: str = ""
    active_subqueries: str = ""
    resolved_subqueries: str = ""
    open_gaps: str = ""
    dossier_context: str = ""
    evidentiary: list[CuratedEvidence] = Field(default_factory=list)
    local_source: str = ""
    topic_briefs_context: str = ""


class ContextManager:
    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._prompt_loader = PromptTemplateLoader(config.prompts_dir, strict_templates=True)

    def _render_bulleted(self, lines: list[str], *, empty_text: str = "- None") -> str:
        if not lines:
            return empty_text
        return "\n".join(lines)

    def _topic_lists(self, state: ResearchState) -> tuple[list[ResearchTopic], list[ResearchTopic]]:
        active = [topic for topic in state["plan"] if topic.status in {TopicStatus.PENDING, TopicStatus.IN_PROGRESS}]
        resolved = [topic for topic in state["plan"] if topic.status == TopicStatus.COMPLETED]
        return active, resolved

    def _render_topics(self, topics: list[ResearchTopic]) -> str:
        return self._render_bulleted([f"- {topic.id}: {topic.question}" for topic in topics])

    def _render_gaps(self, gaps: list[Gap]) -> str:
        return self._render_bulleted([f"- {gap.topic_id}: {gap.description}" for gap in gaps[:5]])

    def _render_coverage_summary(self, state: ResearchState) -> str:
        if not state["plan"]:
            return "- No topics yet."

        evidence_domains: dict[str, Counter[str]] = defaultdict(Counter)
        for evidence in state["curated_evidence"]:
            for source in evidence.sources:
                evidence_domains[evidence.topic_id][extract_domain(source.url)] += 1

        lines: list[str] = []
        for topic in state["plan"]:
            coverage = state["topic_coverage"].get(topic.id)
            accepted = coverage.accepted_evidence_count if coverage else 0
            unique_domains = coverage.unique_domains if coverage else 0
            domain_list = ", ".join(
                f"{domain} ({count})" for domain, count in evidence_domains[topic.id].most_common(3)
            ) or "none"
            lines.append(
                f"- {topic.id} [{topic.status.value}] target={topic.evidence_target}, "
                f"evidence={accepted}, domains={unique_domains} ({domain_list})"
            )
        return "\n".join(lines)

    def _render_source_balance_summary(self, state: ResearchState) -> str:
        if not state["curated_evidence"]:
            return "- No evidence has been accepted yet."

        domains = Counter(
            extract_domain(source.url)
            for evidence in state["curated_evidence"]
            for source in evidence.sources
        )
        total = sum(domains.values())
        top_domains = ", ".join(f"{domain} ({count})" for domain, count in domains.most_common(5))
        return (
            f"- Accepted evidence items: {len(state['curated_evidence'])}\n"
            f"- Unique evidence domains: {len(domains)}\n"
            f"- Top domains: {top_domains}\n"
            f"- Evidence-source references: {total}"
        )

    def _available_prompt_budget(self) -> int:
        reserved_output = min(
            self._config.model.num_predict,
            int(self._config.model.num_ctx * self._config.reporter.output_reserve_ratio),
        )
        return max(0, self._config.model.num_ctx - reserved_output - self._config.reporter.prompt_margin_tokens)

    def _half_prompt_budget(self) -> int:
        return max(1, self._available_prompt_budget() // 2)

    def _build_dossier(self, state: ResearchState) -> str:
        chunks: list[str] = []
        for topic in state["plan"]:
            summary = state["working_dossier"].topic_summaries.get(topic.id, "")
            gap_lines = [
                f"- Gap: {gap.description}"
                for gap in state["open_gaps"]
                if gap.topic_id == topic.id
            ]
            topic_chunks = [f"{topic.id} | {topic.question}", summary, *gap_lines]
            chunks.append("\n".join(chunk for chunk in topic_chunks if chunk).strip())
        return "\n\n".join(chunk for chunk in chunks if chunk)

    def _render_topic_briefs(
        self,
        briefs: dict[str, TopicBrief],
        *,
        budget_tokens: int,
    ) -> str:
        if not briefs or budget_tokens <= 0:
            return ""
        rendered: list[str] = []
        consumed = 0
        for brief in briefs.values():
            block = f"## {brief.question}\n{brief.markdown_brief}".strip()
            block_tokens = estimate_tokens(block)
            if rendered and consumed + block_tokens > budget_tokens:
                break
            if not rendered and block_tokens > budget_tokens:
                rendered.append(block)
                break
            rendered.append(block)
            consumed += block_tokens
        return "\n\n".join(rendered)

    def _context_evidence(
        self,
        state: ResearchState,
        *,
        topic_ids: list[str],
        budget_tokens: int,
    ) -> list[CuratedEvidence]:
        return select_evidence_for_context(
            state["curated_evidence"],
            topic_ids=topic_ids,
            budget_tokens=budget_tokens,
        )

    def synthesis_budget(self, state: ResearchState) -> SynthesisBudget:
        dossier = self._build_dossier(state)
        base_prompt = self._prompt_loader.render(
            "synthesizer",
            {
                "query": state["query"],
                "coverage_summary": "",
                "source_balance_summary": "",
                "open_gaps": "",
                "dossier_context": dossier,
                "topic_briefs_context": "",
                "evidentiary": "",
                "language": self._config.runtime.language,
                "target_words": self._config.reporter.final_report_target_words,
                "format_instructions": "",
            },
        )
        base_prompt_tokens = estimate_tokens(base_prompt.system) + estimate_tokens(base_prompt.human)
        available_prompt_tokens = self._available_prompt_budget()
        candidate_topic_ids = [topic.id for topic in state["plan"] if topic.status == TopicStatus.COMPLETED]
        if not candidate_topic_ids:
            candidate_topic_ids = [
                topic.id
                for topic in state["plan"]
                if topic.status in {TopicStatus.PENDING, TopicStatus.IN_PROGRESS}
            ]
        candidate_evidence = select_evidence_for_context(
            state["curated_evidence"],
            topic_ids=candidate_topic_ids,
            budget_tokens=max(1, self._config.model.num_ctx),
        )
        evidence_budget = max(0, available_prompt_tokens - base_prompt_tokens)
        selected_evidence = select_evidence_for_context(
            candidate_evidence,
            topic_ids=[evidence.topic_id for evidence in candidate_evidence],
            budget_tokens=evidence_budget,
        )
        selected_tokens = sum(evidence.prompt_fit_tokens_estimate for evidence in selected_evidence)
        candidate_tokens = sum(evidence.prompt_fit_tokens_estimate for evidence in candidate_evidence)
        overflow = max(0, base_prompt_tokens + candidate_tokens - available_prompt_tokens)
        reserved_output = min(
            self._config.model.num_predict,
            int(self._config.model.num_ctx * self._config.reporter.output_reserve_ratio),
        )
        return SynthesisBudget(
            context_window_tokens=self._config.model.num_ctx,
            reserved_output_tokens=reserved_output,
            prompt_margin_tokens=self._config.reporter.prompt_margin_tokens,
            base_prompt_tokens=base_prompt_tokens,
            available_prompt_tokens=available_prompt_tokens,
            selected_evidence_tokens=selected_tokens,
            candidate_evidence_tokens=candidate_tokens,
            overflow_tokens=overflow,
            selected_evidence_count=len(selected_evidence),
            candidate_evidence_count=len(candidate_evidence),
            final_context_full=overflow > 0,
        )

    def planner_context(self, state: ResearchState) -> NodeContext:
        active, resolved = self._topic_lists(state)
        return NodeContext(
            query=state["query"],
            has_subqueries=bool(state["plan"]),
            coverage_summary=self._render_coverage_summary(state),
            source_balance_summary=self._render_source_balance_summary(state),
            active_subqueries=self._render_topics(active),
            resolved_subqueries=self._render_topics(resolved),
            open_gaps=self._render_gaps(state["open_gaps"]),
            dossier_context=self._build_dossier(state),
        )

    def extractor_context(self, state: ResearchState, topic: ResearchTopic, local_source: str) -> NodeContext:
        evidence = self._context_evidence(
            state,
            topic_ids=[topic.id],
            budget_tokens=self._half_prompt_budget(),
        )
        return NodeContext(
            query=state["query"],
            active_subqueries=f"- {topic.id}: {topic.question}",
            open_gaps=self._render_gaps([gap for gap in state["open_gaps"] if gap.topic_id == topic.id]),
            evidentiary=evidence,
            local_source=local_source,
        )

    def evaluator_context(self, state: ResearchState) -> NodeContext:
        active, resolved = self._topic_lists(state)
        active_topic_id = state.get("active_topic_id")
        evidence = self._context_evidence(
            state,
            topic_ids=[active_topic_id] if active_topic_id else [],
            budget_tokens=self._half_prompt_budget(),
        )
        return NodeContext(
            query=state["query"],
            coverage_summary=self._render_coverage_summary(state),
            source_balance_summary=self._render_source_balance_summary(state),
            active_subqueries=self._render_topics(active),
            resolved_subqueries=self._render_topics(resolved),
            open_gaps=self._render_gaps(state["open_gaps"]),
            dossier_context=self._build_dossier(state),
            evidentiary=evidence,
        )

    def topic_brief_context(self, state: ResearchState, topic: ResearchTopic) -> NodeContext:
        budget = self._half_prompt_budget()
        evidence = self._context_evidence(
            state,
            topic_ids=[topic.id],
            budget_tokens=budget,
        )
        return NodeContext(
            query=state["query"],
            coverage_summary=self._render_coverage_summary(state),
            source_balance_summary=self._render_source_balance_summary(state),
            active_subqueries=f"- {topic.id}: {topic.question}",
            open_gaps=self._render_gaps([gap for gap in state["open_gaps"] if gap.topic_id == topic.id]),
            dossier_context=self._build_dossier(state),
            evidentiary=evidence,
        )

    def synthesizer_context(self, state: ResearchState) -> NodeContext:
        budget = state.get("synthesis_budget") or self.synthesis_budget(state)
        topic_ids = [topic.id for topic in state["plan"] if topic.status == TopicStatus.COMPLETED]
        if not topic_ids:
            topic_ids = [topic.id for topic in state["plan"]]
        brief_budget = int(budget.available_prompt_tokens * self._config.reporter.topic_brief_budget_ratio)
        topic_briefs_context = self._render_topic_briefs(state.get("topic_briefs", {}), budget_tokens=brief_budget)
        evidence_budget = max(0, budget.available_prompt_tokens - estimate_tokens(topic_briefs_context))
        evidence = self._context_evidence(
            state,
            topic_ids=topic_ids,
            budget_tokens=evidence_budget,
        )
        active, resolved = self._topic_lists(state)
        return NodeContext(
            query=state["query"],
            coverage_summary=self._render_coverage_summary(state),
            source_balance_summary=self._render_source_balance_summary(state),
            active_subqueries=self._render_topics(active),
            resolved_subqueries=self._render_topics(resolved),
            open_gaps=self._render_gaps(state["open_gaps"]),
            dossier_context=self._build_dossier(state),
            evidentiary=evidence,
            topic_briefs_context=topic_briefs_context,
        )

    def debug_state_snapshot(self, state: ResearchState, *, limit: int = 5) -> dict[str, object]:
        return {
            "coverage_summary": self._render_coverage_summary(state),
            "source_balance_summary": self._render_source_balance_summary(state),
            "open_gaps": summarize_gaps(state["open_gaps"], limit=limit),
            "counts": {
                "plan": len(state["plan"]),
                "curated_evidence": len(state["curated_evidence"]),
                "visited_urls": len(state["visited_urls"]),
                "discarded_sources": len(state["discarded_sources"]),
            },
        }


def budget_to_mapping(budget: SynthesisBudget) -> Mapping[str, object]:
    return budget.model_dump(mode="python")
