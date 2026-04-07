"""Prompt context assembly and synthesis budget calculation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from .config import ResearchConfig
from .core.utils import estimate_tokens, extract_domain, select_evidence_for_context, summarize_gaps
from .prompting import PromptTemplateLoader
from .state import CuratedEvidence, Gap, ResearchState, ResearchTopic, SynthesisBudget, TopicStatus


class NodeContext(BaseModel):
    query: str
    today_date: str = ""
    has_subqueries: bool = False
    coverage_summary: str = ""
    source_balance_summary: str = ""
    active_subqueries: str = ""
    resolved_subqueries: str = ""
    open_gaps: str = ""
    dossier_context: str = ""
    evidentiary: list[CuratedEvidence] = Field(default_factory=list)
    local_source: str = ""

    # -- chapter-scoped fields used by micro_planner / auditor / sub_synthesizer --
    chapter_id: str = ""
    chapter_question: str = ""
    chapter_rationale: str = ""
    chapter_criteria: str = ""
    existing_subtopics: str = ""
    subtopics_summary: str = ""
    limitations: str = ""
    audit_attempt: int = 0
    max_audit_attempts: int = 2
    max_chapters: int = 5
    min_chapters: int = 2
    hypotheses: str = ""

    # -- global-synthesizer fields --
    chapters: list[dict[str, object]] = Field(default_factory=list)
    global_limitations: list[str] = Field(default_factory=list)


class ContextManager:
    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._prompt_loader = PromptTemplateLoader(config.prompts_dir, strict_templates=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_bulleted(self, lines: list[str], *, empty_text: str = "- None") -> str:
        if not lines:
            return empty_text
        return "\n".join(lines)

    def _topic_lists(self, state: ResearchState) -> tuple[list[ResearchTopic], list[ResearchTopic]]:
        active = [topic for topic in state["plan"] if topic.status in {TopicStatus.PENDING, TopicStatus.IN_PROGRESS}]
        resolved = [topic for topic in state["plan"] if topic.status == TopicStatus.COMPLETED]
        return active, resolved

    def _chapter_topics(self, state: ResearchState, chapter_id: str) -> list[ResearchTopic]:
        """All topics belonging to a chapter (including the chapter root itself)."""
        return [t for t in state["plan"] if t.chapter_id == chapter_id]

    def _chapter_evidence(self, state: ResearchState, chapter_id: str) -> list[CuratedEvidence]:
        """Curated evidence belonging to a chapter, excluding flushed chapters."""
        flushed = set(state.get("flushed_chapter_ids") or [])
        if chapter_id in flushed:
            return []
        return [e for e in state["curated_evidence"] if e.chapter_id == chapter_id]

    def _unflushed_evidence(self, state: ResearchState) -> list[CuratedEvidence]:
        """All curated evidence whose chapter has not been flushed yet."""
        flushed = set(state.get("flushed_chapter_ids") or [])
        return [e for e in state["curated_evidence"] if e.chapter_id not in flushed]

    def _render_topics(self, topics: list[ResearchTopic]) -> str:
        return self._render_bulleted([f"- {topic.id}: {topic.question}" for topic in topics])

    def _render_gaps(self, gaps: list[Gap]) -> str:
        return self._render_bulleted([f"- {gap.topic_id}: {gap.description}" for gap in gaps[:5]])

    def _render_coverage_summary(self, state: ResearchState, *, chapter_id: str | None = None) -> str:
        topics = state["plan"]
        if chapter_id:
            topics = [t for t in topics if t.chapter_id == chapter_id]
        if not topics:
            return "- No topics yet."

        evidence_domains: dict[str, Counter[str]] = defaultdict(Counter)
        for evidence in state["curated_evidence"]:
            for source in evidence.sources:
                evidence_domains[evidence.topic_id][extract_domain(source.url)] += 1

        lines: list[str] = []
        for topic in topics:
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

    def _build_dossier(self, state: ResearchState, *, chapter_id: str | None = None) -> str:
        topics = state["plan"]
        if chapter_id:
            topics = [t for t in topics if t.chapter_id == chapter_id]
        chunks: list[str] = []
        for topic in topics:
            summary = state["working_dossier"].topic_summaries.get(topic.id, "")
            chunks.append(f"{topic.id} | {topic.question}\n{summary}".strip())
        return "\n\n".join(chunk for chunk in chunks if chunk)

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

    # ------------------------------------------------------------------
    # Public context builders
    # ------------------------------------------------------------------

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def meta_planner_context(self, state: ResearchState) -> NodeContext:
        """Context for the meta-planner: query + existing hypotheses."""
        return NodeContext(
            query=state["query"],
            today_date=self._today(),
            hypotheses="\n".join(f"- {h}" for h in state.get("hypotheses") or []) or "- None",
            max_chapters=self._config.runtime.max_chapters,
            min_chapters=max(1, self._config.runtime.max_chapters // 2),
        )

    def micro_planner_context(self, state: ResearchState, chapter: ResearchTopic) -> NodeContext:
        """Context for the micro-planner: chapter + existing evidence + gaps."""
        chapter_id = chapter.chapter_id or chapter.id
        chapter_topics = self._chapter_topics(state, chapter_id)
        chapter_evidence = self._chapter_evidence(state, chapter_id)
        chapter_gaps = [g for g in state["open_gaps"] if g.topic_id in {t.id for t in chapter_topics}]
        existing = [t for t in chapter_topics if t.depth > 0]
        return NodeContext(
            query=state["query"],
            today_date=self._today(),
            chapter_id=chapter_id,
            chapter_question=chapter.question,
            chapter_rationale=chapter.rationale,
            chapter_criteria="\n".join(f"- {c}" for c in chapter.success_criteria) or "- None",
            existing_subtopics=self._render_topics(existing),
            open_gaps=self._render_gaps(chapter_gaps),
            dossier_context=self._build_dossier(state, chapter_id=chapter_id),
            coverage_summary=self._render_coverage_summary(state, chapter_id=chapter_id),
            evidentiary=chapter_evidence,
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
        """Context for the deterministic evaluator, scoped to the current chapter."""
        chapter_id = state.get("current_chapter_id") or ""
        chapter_topics = self._chapter_topics(state, chapter_id) if chapter_id else state["plan"]
        active = [t for t in chapter_topics if t.status in {TopicStatus.PENDING, TopicStatus.IN_PROGRESS}]
        resolved = [t for t in chapter_topics if t.status == TopicStatus.COMPLETED]
        topic_ids = [t.id for t in chapter_topics]
        evidence = self._context_evidence(
            state,
            topic_ids=topic_ids,
            budget_tokens=self._half_prompt_budget(),
        )
        return NodeContext(
            query=state["query"],
            chapter_id=chapter_id,
            coverage_summary=self._render_coverage_summary(state, chapter_id=chapter_id),
            source_balance_summary=self._render_source_balance_summary(state),
            active_subqueries=self._render_topics(active),
            resolved_subqueries=self._render_topics(resolved),
            open_gaps=self._render_gaps([g for g in state["open_gaps"] if g.topic_id in set(topic_ids)]),
            dossier_context=self._build_dossier(state, chapter_id=chapter_id),
            evidentiary=evidence,
        )

    def auditor_context(self, state: ResearchState, chapter: ResearchTopic) -> NodeContext:
        """Context for the auditor node: chapter evidence + coverage for devil's advocate review."""
        chapter_id = chapter.chapter_id or chapter.id
        chapter_topics = self._chapter_topics(state, chapter_id)
        chapter_evidence = self._chapter_evidence(state, chapter_id)
        topic_ids = {t.id for t in chapter_topics}
        chapter_gaps = [g for g in state["open_gaps"] if g.topic_id in topic_ids]
        subtopics = [t for t in chapter_topics if t.depth > 0]
        audit_attempts = (state.get("topic_audit_attempts") or {}).get(chapter_id, 0)
        return NodeContext(
            query=state["query"],
            chapter_id=chapter_id,
            chapter_question=chapter.question,
            chapter_criteria="\n".join(f"- {c}" for c in chapter.success_criteria) or "- None",
            subtopics_summary=self._render_topics(subtopics),
            coverage_summary=self._render_coverage_summary(state, chapter_id=chapter_id),
            open_gaps=self._render_gaps(chapter_gaps),
            evidentiary=chapter_evidence,
            audit_attempt=audit_attempts + 1,
            max_audit_attempts=self._config.runtime.max_audit_rejections,
        )

    def sub_synthesizer_context(self, state: ResearchState, chapter: ResearchTopic) -> NodeContext:
        """Context for per-chapter synthesis: only chapter evidence within budget."""
        chapter_id = chapter.chapter_id or chapter.id
        chapter_evidence = self._chapter_evidence(state, chapter_id)
        chapter_topics = self._chapter_topics(state, chapter_id)
        topic_ids = [t.id for t in chapter_topics]
        chapter_gaps = [g for g in state["open_gaps"] if g.topic_id in set(topic_ids)]
        limitations = [g.description for g in chapter_gaps if not g.actionable]
        selected = select_evidence_for_context(
            chapter_evidence,
            topic_ids=topic_ids,
            budget_tokens=self._available_prompt_budget(),
        )
        return NodeContext(
            query=state["query"],
            chapter_id=chapter_id,
            chapter_question=chapter.question,
            open_gaps=self._render_gaps(chapter_gaps),
            limitations="\n".join(f"- {lim}" for lim in limitations) or "- None",
            evidentiary=selected,
        )

    def global_synthesizer_context(self, state: ResearchState) -> NodeContext:
        """Context for the final report: chapter drafts + cross-chapter info."""
        drafts = state.get("chapter_drafts") or []
        # Convert ChapterDraft models into plain dicts for the Jinja template
        chapters_data: list[dict[str, object]] = []
        all_evidence_ids: list[str] = []
        for draft in drafts:
            chapters_data.append(draft.model_dump())
            all_evidence_ids.extend(draft.evidence_ids)

        # Collect all evidence across chapters for source building
        all_evidence = [e for e in state["curated_evidence"] if e.evidence_id in set(all_evidence_ids)]

        # Gather cross-chapter limitations
        global_lims: list[str] = []
        for draft in drafts:
            global_lims.extend(draft.limitations)

        return NodeContext(
            query=state["query"],
            chapters=chapters_data,
            global_limitations=global_lims,
            evidentiary=all_evidence,
        )

    def synthesis_budget(self, state: ResearchState) -> SynthesisBudget:
        base_prompt = self._prompt_loader.render(
            "global_synthesizer",
            {
                "query": state["query"],
                "chapters": [],
                "global_limitations": [],
                "language": self._config.runtime.language,
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
