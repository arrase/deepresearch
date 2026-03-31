"""Explicit contracts and registry for internal subagents.

Each subagent declares whether it is deterministic or LLM-based, together with
its inputs and outputs. The goal is to make the architectural choice explicit
instead of letting implementation convenience decide it implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SubagentKind(str, Enum):
    DETERMINISTIC = "deterministic"
    LLM = "llm"


@dataclass(frozen=True)
class SubagentContract:
    name: str
    kind: SubagentKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    rationale: str


SUBAGENT_REGISTRY: dict[str, SubagentContract] = {
    "canonicalize_url": SubagentContract(
        name="canonicalize_url",
        kind=SubagentKind.DETERMINISTIC,
        inputs=("url",),
        outputs=("canonical_url",),
        rationale="URL canonicalization and deduplication are mechanical tasks that are reliable without an LLM.",
    ),
    "candidate_ranker": SubagentContract(
        name="candidate_ranker",
        kind=SubagentKind.DETERMINISTIC,
        inputs=("candidate", "subqueries", "visited_urls"),
        outputs=("score", "reasons"),
        rationale="Initial heuristic ranking can be resolved with transparent and auditable rules.",
    ),
    "planner": SubagentContract(
        name="planner",
        kind=SubagentKind.LLM,
        inputs=("query", "context_window_config"),
        outputs=("subqueries", "search_intents", "hypotheses"),
        rationale="Decomposing open-ended questions and producing an agenda requires semantic interpretation.",
    ),
    "evidence_extractor": SubagentContract(
        name="evidence_extractor",
        kind=SubagentKind.LLM,
        inputs=("source_fragment", "target_subquery", "context"),
        outputs=("atomic_evidence",),
        rationale="Flexible extraction of facts and caveats from natural language requires controlled interpretation.",
    ),
    "coverage_evaluator": SubagentContract(
        name="coverage_evaluator",
        kind=SubagentKind.LLM,
        inputs=("working_dossier", "subqueries", "evidence"),
        outputs=("coverage_assessment", "gaps", "contradictions"),
        rationale="Detecting non-trivial gaps and subtle contradictions can require semantic compression.",
    ),
    "discord_notifier": SubagentContract(
        name="discord_notifier",
        kind=SubagentKind.DETERMINISTIC,
        inputs=("config", "report"),
        outputs=("success",),
        rationale="Sending a notification to an external API is a mechanical side-effect.",
    ),
}
