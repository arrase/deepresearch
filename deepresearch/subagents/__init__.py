"""Contratos y registro explicito de subagentes internos.

Cada subagente declara si es determinista o basado en LLM, junto con sus
entradas y salidas. El objetivo es hacer visible por que una tarea usa o no un
modelo y evitar que la comodidad de implementacion sustituya una decision de
arquitectura.
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
        rationale="La canonizacion y deduplicacion de URLs es una tarea mecanica y fiable sin LLM.",
    ),
    "candidate_ranker": SubagentContract(
        name="candidate_ranker",
        kind=SubagentKind.DETERMINISTIC,
        inputs=("candidate", "subqueries", "visited_urls"),
        outputs=("score", "reasons"),
        rationale="El ranking heuristico inicial puede resolverse con reglas transparentes y auditables.",
    ),
    "planner": SubagentContract(
        name="planner",
        kind=SubagentKind.LLM,
        inputs=("query", "context_window_config"),
        outputs=("subqueries", "search_intents", "hypotheses"),
        rationale="Descomponer preguntas abiertas y generar agenda requiere interpretacion semantica.",
    ),
    "evidence_extractor": SubagentContract(
        name="evidence_extractor",
        kind=SubagentKind.LLM,
        inputs=("source_fragment", "target_subquery", "context"),
        outputs=("atomic_evidence",),
        rationale="La extraccion flexible de hechos y reservas desde texto natural requiere interpretacion controlada.",
    ),
    "coverage_evaluator": SubagentContract(
        name="coverage_evaluator",
        kind=SubagentKind.LLM,
        inputs=("working_dossier", "subqueries", "evidence"),
        outputs=("coverage_assessment", "gaps", "contradictions"),
        rationale="La deteccion de huecos no triviales y contradicciones sutiles puede necesitar compresion semantica.",
    ),
}
