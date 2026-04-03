"""Hierarchical context manager."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping

from pydantic import BaseModel, Field

from .config import ResearchConfig
from .core.utils import (
    estimate_tokens,
    extract_domain,
    select_evidence_for_context,
    summarize_gaps,
    total_evidence_tokens,
)
from .prompting import PromptTemplateLoader
from .state import AtomicEvidence, Gap, ResearchState, Subquery


class NodeContext(BaseModel):
    query: str
    has_subqueries: bool = False
    coverage_summary: str = ""
    source_balance_summary: str = ""
    active_subqueries: str = ""
    resolved_subqueries: str = ""
    open_gaps: str = ""
    dossier_context: str = ""
    evidentiary: list[AtomicEvidence] = Field(default_factory=list)
    local_source: str = ""

class ContextManager:
    def __init__(self, config: ResearchConfig) -> None:
        self._config = config
        self._prompt_loader = PromptTemplateLoader(config.prompts_dir, strict_templates=True)

    def _render_sq(self, sqs: list[Subquery]) -> str:
        return "\n".join(f"- {s.id}: {s.question}" for s in sqs) or "- None"

    def _render_gaps(self, gaps: list[Gap]) -> str:
        return "\n".join(f"- {g.subquery_id}: {g.description}" for g in gaps[:5]) or "- None"

    def _render_coverage_summary(self, state: ResearchState) -> str:
        subqueries = [*state["active_subqueries"], *state["resolved_subqueries"]]
        if not subqueries:
            return "- No subqueries yet."

        evidence_by_subquery = Counter(e.subquery_id for e in state["atomic_evidence"])
        domain_counts_by_subquery: dict[str, Counter[str]] = defaultdict(Counter)
        for evidence in state["atomic_evidence"]:
            domain_counts_by_subquery[evidence.subquery_id][extract_domain(evidence.source_url)] += 1

        lines = []
        for subquery in subqueries:
            evidence_count = evidence_by_subquery[subquery.id]
            domains = domain_counts_by_subquery[subquery.id]
            domain_list = ", ".join(f"{domain} ({count})" for domain, count in domains.most_common(3)) or "none"
            status = "resolved" if any(sq.id == subquery.id for sq in state["resolved_subqueries"]) else "active"

            flags: list[str] = []
            if evidence_count == 0:
                flags.append("no evidence yet")
            if 0 < evidence_count < subquery.evidence_target:
                flags.append(f"needs {subquery.evidence_target - evidence_count} more evidence items")
            if evidence_count >= 2 and len(domains) <= 1:
                flags.append("low source diversity")

            lines.append(
                f"- {subquery.id} [{status}] target={subquery.evidence_target}, "
                f"evidence={evidence_count}, domains={len(domains)} ({domain_list}); "
                f"signals: {', '.join(flags) if flags else 'coverage looks acceptable'}"
            )
        return "\n".join(lines)

    def _render_source_balance_summary(self, state: ResearchState) -> str:
        if not state["atomic_evidence"]:
            return "- No evidence has been accepted yet."

        domains = Counter(extract_domain(e.source_url) for e in state["atomic_evidence"])
        total = sum(domains.values())
        top_domain, top_count = domains.most_common(1)[0]
        concentration = top_count / total if total else 0.0
        top_domains = ", ".join(f"{domain} ({count})" for domain, count in domains.most_common(5))

        signals: list[str] = []
        if concentration >= 0.6:
            signals.append(f"evidence is highly concentrated in {top_domain}")
        if len(domains) <= 1 and total >= 2:
            signals.append("cross-source validation is weak")
        if len(state["resolved_subqueries"]) < len(state["active_subqueries"]) and total > 0:
            signals.append("some active subqueries may still be thinly evidenced")

        return (
            f"- Accepted evidence items: {total}\n"
            f"- Unique evidence domains: {len(domains)}\n"
            f"- Top domains: {top_domains}\n"
            f"- Coverage risk signals: {', '.join(signals) if signals else 'no major source-balance warning'}"
        )

    def _available_prompt_budget(self) -> int:
        reserved_output = min(
            self._config.model.num_predict,
            int(self._config.model.num_ctx * self._config.runtime.synthesizer_output_reserve_ratio),
        )
        return max(0, self._config.model.num_ctx - reserved_output - self._config.runtime.synthesizer_prompt_margin)

    def _select_evidence_until_budget(self, evidence: list[AtomicEvidence], budget_tokens: int) -> list[AtomicEvidence]:
        selected: list[AtomicEvidence] = []
        consumed = 0
        for item in sorted(evidence, key=lambda x: x.relevance_score, reverse=True):
            tokens = estimate_tokens(item.claim + item.summary + item.quotation)
            if consumed + tokens > budget_tokens:
                break
            selected.append(item)
            consumed += tokens
        return selected

    def _build_synthesizer_dossier(self, state: ResearchState) -> str:
        return "\n\n".join(
            f"{s.id} | {s.question}\n{state['working_dossier'].subquery_summaries.get(s.id, '')}"
            for s in [*state["resolved_subqueries"], *state["active_subqueries"]]
        )

    def synthesis_budget(self, state: ResearchState) -> dict[str, int | bool | str | None]:
        subquery_ids = [s.id for s in state["resolved_subqueries"] or state["active_subqueries"]]
        candidate_evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=subquery_ids,
            budget_tokens=max(1, self._config.model.num_ctx),
        )
        dossier = self._build_synthesizer_dossier(state)
        base_prompt = self._prompt_loader.render(
            "synthesizer",
            {
                "query": state["query"],
                "dossier_context": dossier,
                "evidentiary": "",
                "language": self._config.runtime.language,
                "format_instructions": "",
            },
        )
        base_prompt_tokens = estimate_tokens(base_prompt.system) + estimate_tokens(base_prompt.human)
        available_prompt_tokens = self._available_prompt_budget()
        evidence_budget = max(0, available_prompt_tokens - base_prompt_tokens)
        selected_evidence = self._select_evidence_until_budget(candidate_evidence, evidence_budget)
        selected_evidence_tokens = total_evidence_tokens(selected_evidence)
        total_candidate_evidence_tokens = total_evidence_tokens(candidate_evidence)
        overflow_tokens = max(0, base_prompt_tokens + total_candidate_evidence_tokens - available_prompt_tokens)
        context_full = bool(candidate_evidence) and (
            overflow_tokens > 0 or len(selected_evidence) < len(candidate_evidence)
        )
        reserved_output = min(
            self._config.model.num_predict,
            int(self._config.model.num_ctx * self._config.runtime.synthesizer_output_reserve_ratio),
        )
        return {
            "context_window_tokens": self._config.model.num_ctx,
            "reserved_output_tokens": reserved_output,
            "prompt_margin_tokens": self._config.runtime.synthesizer_prompt_margin,
            "available_prompt_tokens": available_prompt_tokens,
            "base_prompt_tokens": base_prompt_tokens,
            "evidence_budget_tokens": evidence_budget,
            "selected_evidence_tokens": selected_evidence_tokens,
            "candidate_evidence_tokens": total_candidate_evidence_tokens,
            "overflow_tokens": overflow_tokens,
            "selected_evidence_count": len(selected_evidence),
            "candidate_evidence_count": len(candidate_evidence),
            "final_context_full": context_full,
            "stop_reason": "final_context_full" if context_full else None,
        }

    def planner_context(self, state: ResearchState) -> NodeContext:
        has = bool(state["active_subqueries"] or state["resolved_subqueries"])
        dossier = "\n".join(
            f"Subquery {sid}: {summary}"
            for sid, summary in state["working_dossier"].subquery_summaries.items()
        )
        return NodeContext(
            query=state["query"],
            has_subqueries=has,
            coverage_summary=self._render_coverage_summary(state),
            source_balance_summary=self._render_source_balance_summary(state),
            active_subqueries=self._render_sq(state["active_subqueries"]),
            resolved_subqueries=self._render_sq(state["resolved_subqueries"]),
            open_gaps=self._render_gaps(state["open_gaps"]),
            dossier_context=dossier,
        )

    def extractor_context(self, state: ResearchState, targets: list[str], local_source: str) -> NodeContext:
        budget_tokens = max(1, self._available_prompt_budget() // 2)
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=targets,
            budget_tokens=budget_tokens,
        )
        return NodeContext(
            query=state["query"],
            active_subqueries=self._render_sq(state["active_subqueries"]),
            open_gaps=self._render_gaps(state["open_gaps"]),
            evidentiary=evidence,
            local_source=local_source,
        )

    def evaluator_context(self, state: ResearchState) -> NodeContext:
        budget_tokens = max(1, self._available_prompt_budget() // 2)
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=[s.id for s in state["active_subqueries"]],
            budget_tokens=budget_tokens,
        )
        return NodeContext(
            query=state["query"],
            coverage_summary=self._render_coverage_summary(state),
            source_balance_summary=self._render_source_balance_summary(state),
            active_subqueries=self._render_sq(state["active_subqueries"]),
            resolved_subqueries=self._render_sq(state["resolved_subqueries"]),
            open_gaps=self._render_gaps(state["open_gaps"]),
            evidentiary=evidence,
        )

    def synthesizer_context(self, state: ResearchState) -> NodeContext:
        budget = self.synthesis_budget(state)
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=[s.id for s in state["resolved_subqueries"] or state["active_subqueries"]],
            budget_tokens=_budget_int(budget, "evidence_budget_tokens", default=0),
        )
        dossier = self._build_synthesizer_dossier(state)
        return NodeContext(query=state["query"], dossier_context=dossier, evidentiary=evidence)

    def debug_state_snapshot(self, state: ResearchState, *, limit: int = 5) -> dict[str, object]:
        dossier = state["working_dossier"]
        return {
            "coverage_summary": self._render_coverage_summary(state),
            "source_balance_summary": self._render_source_balance_summary(state),
            "open_gaps": summarize_gaps(state["open_gaps"], limit=limit),
            "working_dossier": {
                "subquery_summaries": [
                    {"subquery_id": sid, "summary": summary}
                    for sid, summary in list(dossier.subquery_summaries.items())[:limit]
                ],
                "key_points": dossier.key_points[:limit],
                "source_summaries": [
                    {"url": url, "summary": summary}
                    for url, summary in list(dossier.source_summaries.items())[:limit]
                ],
                "updated_at": dossier.updated_at,
            },
            "counts": {
                "active_subqueries": len(state["active_subqueries"]),
                "resolved_subqueries": len(state["resolved_subqueries"]),
                "accepted_evidence": len(state["atomic_evidence"]),
                "visited_urls": len(state["visited_urls"]),
                "discarded_sources": len(state["discarded_sources"]),
                "useful_sources_count": state["useful_sources_count"],
            },
        }


def _budget_int(
    budget: Mapping[str, int | bool | str | None],
    key: str,
    *,
    default: int,
) -> int:
    value = budget.get(key)
    return int(value) if isinstance(value, (int, str)) else default
