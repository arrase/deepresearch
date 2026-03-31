"""Research graph nodes and shared runtime dependencies."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .config import ResearchConfig
from .context_manager import ContextManager
from .state import (
    AtomicEvidence,
    BrowserPageStatus,
    BrowserResult,
    ConfidenceLevel,
    DiscardedSource,
    FinalReport,
    Gap,
    GapSeverity,
    ResearchState,
    SearchCandidate,
    SourceDiscardReason,
    SourceVisit,
    Subquery,
)
from .subagents.deterministic import (
    build_report_sources,
    compute_minimum_coverage,
    deduplicate_candidates,
    deduplicate_evidence,
    extract_domain,
    render_markdown_report,
    score_candidate,
    select_relevant_chunks,
    short_excerpt,
    split_text,
    update_working_dossier,
)
from .telemetry import TelemetryRecorder


@dataclass
class ResearchRuntime:
    config: ResearchConfig
    context_manager: ContextManager
    llm_workers: object
    browser: object
    search_client: object
    telemetry: TelemetryRecorder


class ResearchNodes:
    def __init__(self, runtime: ResearchRuntime) -> None:
        self._runtime = runtime

    def _start_event(self, state: ResearchState, stage: str, message: str, **payload: object) -> list:
        event = self._runtime.telemetry.record(stage, message, **payload)
        return [*state["telemetry"], event]

    def node_planner(self, state: ResearchState) -> dict:
        telemetry = self._start_event(
            state,
            "planner",
            "Starting planning with the LLM",
            query=state["query"],
        )
        context = self._runtime.context_manager.planner_context(state)
        payload = self._runtime.llm_workers.plan_research(context)
        
        # Deduplicate and merge subqueries
        existing_ids = {sq.id for sq in [*state["active_subqueries"], *state["resolved_subqueries"]]}
        merged_subqueries = list(state["active_subqueries"])
        added_count = 0
        for nsq in payload.subqueries:
            if nsq.id not in existing_ids:
                merged_subqueries.append(nsq)
                added_count += 1
        
        # Merge search intents and hypotheses
        merged_intents = [*state["search_intents"], *payload.search_intents]
        merged_hypotheses = list(set([*state["hypotheses"], *payload.hypotheses]))

        event = self._runtime.telemetry.record(
            "planner",
            "Research agenda updated",
            added_subqueries=added_count,
            total_active=len(merged_subqueries),
            search_intents=len(merged_intents),
        )
        return {
            "active_subqueries": merged_subqueries,
            "search_intents": merged_intents,
            "hypotheses": merged_hypotheses,
            "telemetry": [*telemetry, event],
        }

    def node_source_manager(self, state: ResearchState) -> dict:
        if state["search_queue"]:
            next_candidate = state["search_queue"][0]
            event = self._runtime.telemetry.record(
                "source_manager",
                "Reusing an already prioritized candidate",
                url=next_candidate.url,
            )
            return {
                "search_queue": state["search_queue"][1:],
                "current_candidate": next_candidate,
                "iteration": state["iteration"] + 1,
                "telemetry": [*state["telemetry"], event],
            }

        candidate_queries: list[str] = []
        # Prioritize open gaps (newest first)
        for gap in reversed(state["open_gaps"]):
            for query in gap.suggested_queries:
                if query not in state["completed_search_queries"] and query not in candidate_queries:
                    candidate_queries.append(query)
        
        # Then newer search intents from iterative planning (deep-dives)
        for intent in reversed(state["search_intents"]):
            if intent.query not in state["completed_search_queries"] and intent.query not in candidate_queries:
                candidate_queries.append(intent.query)
        
        # Fallback to active subqueries
        if not candidate_queries:
            for subquery in reversed(state["active_subqueries"]):
                if subquery.question not in state["completed_search_queries"] and subquery.question not in candidate_queries:
                    candidate_queries.append(subquery.question)

        candidate_queries = candidate_queries[: self._runtime.config.search.max_queries_per_cycle]
        telemetry = self._start_event(
            state,
            "source_manager",
            "Searching for and prioritizing new sources",
            queries=candidate_queries,
        )
        raw_candidates: list[SearchCandidate] = []
        completed = [*state["completed_search_queries"]]
        for query in candidate_queries:
            self._runtime.telemetry.record(
                "source_manager",
                "Querying search backend",
                query=query,
                backend=self._runtime.config.search.backend,
            )
            try:
                results = self._runtime.search_client.search(
                    query,
                    max_results=self._runtime.config.search.results_per_query,
                )
            except Exception as exc:  # noqa: BLE001
                event = self._runtime.telemetry.record(
                    "source_manager",
                    "Search backend failed",
                    query=query,
                    error=str(exc),
                )
                return {
                    "completed_search_queries": [*completed, query],
                    "telemetry": [*telemetry, event],
                    "fallback_reason": state["fallback_reason"] or "search_backend_failure",
                }
            completed.append(query)
            for candidate in results:
                if not candidate.subquery_ids:
                    candidate.subquery_ids = self._match_subqueries(candidate, state["active_subqueries"])
                candidate.domain = candidate.domain or extract_domain(candidate.url)
                raw_candidates.append(candidate)

        deduped, discarded = deduplicate_candidates(raw_candidates, visited_urls=state["visited_urls"])
        discarded_sources = [*state["discarded_sources"]]
        for url, reason, note in discarded:
            discarded_sources.append(DiscardedSource(url=url, reason=reason, note=note))

        domain_counts = Counter(entry.domain for entry in state["search_queue"])
        ranked = [
            score_candidate(
                candidate,
                active_subqueries=state["active_subqueries"],
                visited_urls=state["visited_urls"],
                domain_counts=domain_counts,
            )
            for candidate in deduped
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        if not ranked:
            bootstrap = self._bootstrap_candidates(state)
            if bootstrap:
                bootstrap, bootstrap_discarded = deduplicate_candidates(bootstrap, visited_urls=state["visited_urls"])
                for url, reason, note in bootstrap_discarded:
                    discarded_sources.append(DiscardedSource(url=url, reason=reason, note=note))
                bootstrap_ranked = [
                    score_candidate(
                        candidate,
                        active_subqueries=state["active_subqueries"],
                        visited_urls=state["visited_urls"],
                        domain_counts=domain_counts,
                    )
                    for candidate in bootstrap
                ]
                bootstrap_ranked.sort(key=lambda item: item.score, reverse=True)
                if bootstrap_ranked:
                    next_candidate = bootstrap_ranked[0]
                    event = self._runtime.telemetry.record(
                        "source_manager",
                        "Activating authoritative bootstrap sources",
                        selected=next_candidate.url,
                        candidates=len(bootstrap_ranked),
                    )
                    return {
                        "completed_search_queries": completed,
                        "discarded_sources": discarded_sources,
                        "search_queue": bootstrap_ranked[1:],
                        "current_candidate": next_candidate,
                        "iteration": state["iteration"] + 1,
                        "telemetry": [*telemetry, event],
                    }
            fallback_reason = state["fallback_reason"] or "no_actionable_sources"
            event = self._runtime.telemetry.record(
                "source_manager",
                "No new actionable sources are available",
                queries=candidate_queries,
            )
            return {
                "completed_search_queries": completed,
                "discarded_sources": discarded_sources,
                "current_candidate": None,
                "iteration": state["iteration"] + 1,
                "fallback_reason": fallback_reason,
                "telemetry": [*telemetry, event],
            }

        next_candidate = ranked[0]
        event = self._runtime.telemetry.record(
            "source_manager",
            "Sources discovered and prioritized",
            queries=candidate_queries,
            selected=next_candidate.url,
            candidates=len(ranked),
        )
        return {
            "completed_search_queries": completed,
            "discarded_sources": discarded_sources,
            "search_queue": ranked[1:],
            "current_candidate": next_candidate,
            "iteration": state["iteration"] + 1,
            "telemetry": [*telemetry, event],
        }

    def node_browser(self, state: ResearchState) -> dict:
        candidate = state.get("current_candidate")
        if candidate is None:
            result = BrowserResult(
                url="",
                status=BrowserPageStatus.TERMINAL_ERROR,
                error="No actionable candidate is available",
            )
            event = self._runtime.telemetry.record(
                "browser",
                "Skipping navigation because there is no current candidate",
            )
            return {
                "current_browser_result": result,
                "telemetry": [*state["telemetry"], event],
            }

        telemetry = self._start_event(
            state,
            "browser",
            "Navigating with Lightpanda",
            url=candidate.url,
        )
        result = self._runtime.browser.fetch(candidate.url)
        visited = dict(state["visited_urls"])
        resolved_title = candidate.title or result.title
        visited[result.url] = SourceVisit(
            url=result.url,
            final_url=result.final_url,
            title=resolved_title,
            status=result.status,
            content_excerpt=result.excerpt,
            error=result.error,
            candidate_subquery_ids=candidate.subquery_ids,
            diagnostics=result.diagnostics,
        )
        discarded_sources = [*state["discarded_sources"]]
        if result.status not in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
            reason_map = {
                BrowserPageStatus.BLOCKED: SourceDiscardReason.BLOCKED,
                BrowserPageStatus.EMPTY: SourceDiscardReason.EMPTY,
                BrowserPageStatus.RECOVERABLE_ERROR: SourceDiscardReason.TECHNICAL_ERROR,
                BrowserPageStatus.TERMINAL_ERROR: SourceDiscardReason.TECHNICAL_ERROR,
            }
            discarded_sources.append(
                DiscardedSource(
                    url=candidate.url,
                    reason=reason_map.get(result.status, SourceDiscardReason.LOW_VALUE),
                    note=result.error or result.status.value,
                )
            )
        event = self._runtime.telemetry.record(
            "browser",
            "Navigation completed",
            url=candidate.url,
            status=result.status.value,
        )
        return {
            "visited_urls": visited,
            "discarded_sources": discarded_sources,
            "current_browser_result": result,
            "telemetry": [*telemetry, event],
        }

    def node_extractor(self, state: ResearchState) -> dict:
        browser_result = state.get("current_browser_result")
        candidate = state.get("current_candidate")
        if browser_result is None or candidate is None:
            return {"latest_evidence": []}
        if browser_result.status not in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
            return {"latest_evidence": []}

        target_subquery_ids = candidate.subquery_ids or [state["active_subqueries"][0].id]
        query_terms: list[str] = []
        for subquery in state["active_subqueries"]:
            if subquery.id in target_subquery_ids:
                query_terms.extend(subquery.search_terms or [subquery.question])
        chunks = split_text(browser_result.content)
        selected_chunks = select_relevant_chunks(chunks, query_terms=query_terms, max_chunks=10)
        local_source = "\n\n".join(selected_chunks)[: self._runtime.config.browser.max_content_chars]
        telemetry = self._start_event(
            state,
            "extractor",
            "Extracting evidence with the LLM",
            url=browser_result.url,
            chunks=len(selected_chunks),
        )
        context = self._runtime.context_manager.extractor_context(
            state,
            target_subquery_ids=target_subquery_ids,
            local_source=local_source,
        )
        payload = self._runtime.llm_workers.extract_evidence(context)
        latest_evidence: list[AtomicEvidence] = []
        primary_subquery_id = target_subquery_ids[0]
        for item in payload.evidences:
            latest_evidence.append(
                AtomicEvidence(
                    subquery_id=primary_subquery_id,
                    source_url=browser_result.final_url or browser_result.url,
                    source_title=candidate.title or browser_result.title or short_excerpt(local_source, 80),
                    summary=item.summary,
                    claim=item.claim,
                    quotation=item.quotation,
                    citation_locator=item.citation_locator,
                    relevance_score=item.relevance_score,
                    confidence=item.confidence,
                    caveats=item.caveats,
                    tags=item.tags,
                )
            )
        event = self._runtime.telemetry.record(
            "extractor",
            "Evidence extraction completed",
            url=browser_result.url,
            evidence=len(latest_evidence),
        )
        return {
            "latest_evidence": latest_evidence,
            "telemetry": [*telemetry, event],
        }

    def node_context_manager(self, state: ResearchState) -> dict:
        browser_result = state.get("current_browser_result")
        latest = state.get("latest_evidence", [])
        telemetry = self._start_event(
            state,
            "context_manager",
            "Integrating evidence into the dossier",
            new_evidence=len(latest),
        )
        accepted = deduplicate_evidence(state["atomic_evidence"], latest)
        updated_evidence = [*state["atomic_evidence"], *accepted]
        dossier = update_working_dossier(
            state["working_dossier"],
            evidence=accepted,
            source_url=browser_result.url if browser_result else None,
            source_title=browser_result.title if browser_result else None,
        )
        discarded_sources = [*state["discarded_sources"]]
        if browser_result and not accepted and browser_result.status in {BrowserPageStatus.USEFUL, BrowserPageStatus.PARTIAL}:
            discarded_sources.append(
                DiscardedSource(
                    url=browser_result.url,
                    reason=SourceDiscardReason.NO_EVIDENCE,
                    note="The source loaded successfully but produced no acceptable evidence",
                )
            )
        event = self._runtime.telemetry.record(
            "context_manager",
            "Dossier updated",
            accepted_evidence=len(accepted),
            total_evidence=len(updated_evidence),
        )
        return {
            "atomic_evidence": updated_evidence,
            "working_dossier": dossier,
            "discarded_sources": discarded_sources,
            "latest_evidence": accepted,
            "telemetry": [*telemetry, event],
        }

    def node_evaluator(self, state: ResearchState) -> dict:
        telemetry = self._start_event(
            state,
            "evaluator",
            "Evaluating coverage and stop criteria",
            active_subqueries=len(state["active_subqueries"]),
            evidence=len(state["atomic_evidence"]),
        )
        deterministic_resolved_ids, deterministic_gaps = compute_minimum_coverage(
            active_subqueries=state["active_subqueries"],
            evidence=state["atomic_evidence"],
        )
        context = self._runtime.context_manager.evaluator_context(state)
        semantic = self._runtime.llm_workers.evaluate_coverage(context)
        resolved_ids = set(deterministic_resolved_ids)
        evidence_count_by_subquery = Counter(item.subquery_id for item in state["atomic_evidence"])
        for subquery_id in semantic.resolved_subquery_ids:
            if evidence_count_by_subquery[subquery_id] >= 1:
                resolved_ids.add(subquery_id)

        remaining_active: list[Subquery] = []
        newly_resolved: list[Subquery] = []
        for subquery in state["active_subqueries"]:
            if subquery.id in resolved_ids:
                newly_resolved.append(subquery.model_copy(update={"status": "resolved"}))
            else:
                remaining_active.append(subquery)

        gap_index: dict[tuple[str, str], Gap] = {}
        for gap in [*deterministic_gaps, *semantic.open_gaps]:
            gap_index[(gap.subquery_id, gap.description)] = gap
        open_gaps = list(gap_index.values())

        # Phase 3: Dynamic evidence target adjustment
        contradiction_subquery_ids = set()
        for contradiction in semantic.contradictions:
            for ev_id in contradiction.evidence_ids:
                for ev in state["atomic_evidence"]:
                    if ev.id == ev_id:
                        contradiction_subquery_ids.add(ev.subquery_id)
                        break
        
        gap_subquery_ids = {gap.subquery_id for gap in open_gaps if gap.severity in {GapSeverity.HIGH, GapSeverity.CRITICAL}}
        
        adjusted_active = []
        for subquery in remaining_active:
            new_target = subquery.evidence_target
            if subquery.id in contradiction_subquery_ids or subquery.id in gap_subquery_ids:
                new_target = min(subquery.evidence_target + 2, 10)
                self._runtime.telemetry.record(
                    "evaluator",
                    "Increasing evidence target due to uncertainty or contradiction",
                    subquery_id=subquery.id,
                    new_target=new_target,
                )
            adjusted_active.append(subquery.model_copy(update={"evidence_target": new_target}))
        remaining_active = adjusted_active

        fallback_reason = state["fallback_reason"]
        if state["iteration"] >= state["max_iterations"] and fallback_reason is None:
            fallback_reason = "max_iterations_reached"

        is_sufficient = False
        resolved_total = len(state["resolved_subqueries"]) + len(newly_resolved)
        minimum_report_evidence = 1 if resolved_total <= 1 else max(2, resolved_total)
        
        # Stop criteria logic:
        # 1. LLM evaluator explicitly signals sufficiency and no active subqueries remain.
        if semantic.is_sufficient and not remaining_active and len(state["atomic_evidence"]) >= minimum_report_evidence:
            is_sufficient = True
        # 2. We have no active subqueries and no open gaps (deterministic sufficiency).
        elif not remaining_active and not open_gaps and len(state["atomic_evidence"]) >= minimum_report_evidence:
            is_sufficient = True
        # 3. We hit a fallback (search failure, no more sources) and have at least SOME evidence.
        elif fallback_reason is not None and len(state["atomic_evidence"]) >= 1:
            is_sufficient = True
        # 4. We hit a fallback and have no evidence, but we've tried enough or it's a terminal search failure.
        elif fallback_reason in ("search_backend_failure", "no_actionable_sources") and state["iteration"] >= 2:
            is_sufficient = True
        # 5. Hard limit on iterations.
        elif state["iteration"] >= state["max_iterations"]:
            is_sufficient = True

        event = self._runtime.telemetry.record(
            "evaluator",
            "Coverage evaluated",
            resolved=len(newly_resolved),
            remaining=len(remaining_active),
            contradictions=len(semantic.contradictions),
            fallback_reason=fallback_reason,
            is_sufficient=is_sufficient,
        )
        return {
            "active_subqueries": remaining_active,
            "resolved_subqueries": [*state["resolved_subqueries"], *newly_resolved],
            "open_gaps": open_gaps,
            "contradictions": semantic.contradictions,
            "is_sufficient": is_sufficient,
            "fallback_reason": fallback_reason,
            "current_candidate": None,
            "telemetry": [*telemetry, event],
        }

    def node_synthesizer(self, state: ResearchState) -> dict:
        telemetry = self._start_event(
            state,
            "synthesizer",
            "Synthesizing final report",
            evidence=len(state["atomic_evidence"]),
            resolved=len(state["resolved_subqueries"]),
        )
        context = self._runtime.context_manager.synthesizer_context(state)
        try:
            report = self._runtime.llm_workers.synthesize_report(context, query=state["query"])
        except Exception:  # noqa: BLE001
            report = self._fallback_report(state)
        if not report.markdown_report:
            report.markdown_report = render_markdown_report(report)
        event = self._runtime.telemetry.record(
            "synthesizer",
            "Final report generated",
            evidence=len(state["atomic_evidence"]),
            sources=len(report.cited_sources),
        )
        return {
            "final_report": report,
            "telemetry": [*telemetry, event],
        }

    def _match_subqueries(
        self,
        candidate: SearchCandidate,
        active_subqueries: list[Subquery],
    ) -> list[str]:
        haystack = f"{candidate.title} {candidate.snippet}".lower()
        matches = []
        for subquery in active_subqueries:
            terms = [term.lower() for term in subquery.search_terms or [subquery.question]]
            if any(term in haystack for term in terms[:6]):
                matches.append(subquery.id)
        return matches

    def _fallback_report(self, state: ResearchState) -> FinalReport:
        evidence = state["atomic_evidence"]
        answer = state["working_dossier"].global_summary or "The system could not gather enough evidence to answer with verifiable support. Review fallback_reason and telemetry for diagnostics."
        confidence = ConfidenceLevel.MEDIUM if len(evidence) >= 2 else ConfidenceLevel.LOW
        sections = []
        relevant_subqueries = [*state["resolved_subqueries"], *state["active_subqueries"]]
        for subquery in relevant_subqueries[:4]:
            section_evidence = [item for item in evidence if item.subquery_id == subquery.id]
            if not section_evidence:
                continue
            sections.append(
                {
                    "title": subquery.question,
                    "summary": state["working_dossier"].subquery_summaries.get(subquery.id, "") or section_evidence[0].summary,
                    "body": "\n".join(f"- {item.claim}" for item in section_evidence[:4]),
                    "evidence_ids": [item.id for item in section_evidence[:4]],
                    "subquery_ids": [subquery.id],
                }
            )
        report = FinalReport(
            query=state["query"],
            executive_answer=answer,
            key_findings=[item.claim for item in evidence[:5]],
            sections=sections,
            confidence=confidence,
            reservations=[gap.description for gap in state["open_gaps"][:5]],
            open_gaps=[gap.description for gap in state["open_gaps"][:5]],
            cited_sources=build_report_sources(evidence),
            evidence_ids=[item.id for item in evidence],
            stop_reason=state["fallback_reason"],
        )
        report.markdown_report = render_markdown_report(report)
        return report

    def _bootstrap_candidates(self, state: ResearchState) -> list[SearchCandidate]:
        haystacks = [state["query"].lower()]
        haystacks.extend(item.question.lower() for item in state["active_subqueries"])
        haystacks.extend(item.query.lower() for item in state["search_intents"])
        if not any("lightpanda" in item for item in haystacks):
            return []
        subquery_ids = [item.id for item in state["active_subqueries"]]
        return [
            SearchCandidate(
                url="https://lightpanda.io/docs/index",
                title="Lightpanda Documentation Index",
                snippet="Official Lightpanda documentation.",
                domain="lightpanda.io",
                source_type="bootstrap",
                score=5.0,
                reasons=["official_docs_bootstrap"],
                subquery_ids=subquery_ids,
                discovered_via="bootstrap",
            ),
            SearchCandidate(
                url="https://github.com/lightpanda-io/browser",
                title="Lightpanda Browser GitHub Repository",
                snippet="Official Lightpanda source repository and README.",
                domain="github.com",
                source_type="bootstrap",
                score=4.5,
                reasons=["official_repo_bootstrap"],
                subquery_ids=subquery_ids,
                discovered_via="bootstrap",
            ),
            SearchCandidate(
                url="https://lightpanda.io/",
                title="Lightpanda Homepage",
                snippet="Official Lightpanda homepage.",
                domain="lightpanda.io",
                source_type="bootstrap",
                score=4.0,
                reasons=["official_homepage_bootstrap"],
                subquery_ids=subquery_ids,
                discovered_via="bootstrap",
            ),
        ]
