"""Gestor de contexto jerarquico.

Cada nodo consume una proyeccion minima del estado. La seleccion es
determinista primero: subpreguntas activas, huecos accionables y evidencias mas
 relevantes. Cuando el dossier crezca, el runtime podra aplicar compresion
semantica adicional sin cambiar este contrato.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from config import ResearchConfig
from state import AtomicEvidence, Gap, ResearchState, Subquery
from subagents.deterministic import estimate_tokens, select_evidence_for_context


class NodeContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    permanent: str
    strategic: str
    operational: str
    evidentiary: list[AtomicEvidence] = Field(default_factory=list)
    local_source: str = ""


class ContextManager:
    def __init__(self, config: ResearchConfig) -> None:
        self._config = config

    def _render_subqueries(self, subqueries: list[Subquery]) -> str:
        return "\n".join(
            f"- {item.id}: {item.question} | criterios={'; '.join(item.success_criteria[:3])}"
            for item in subqueries
        )

    def _render_gaps(self, gaps: list[Gap]) -> str:
        return "\n".join(
            f"- {gap.subquery_id}: {gap.description} | consultas={', '.join(gap.suggested_queries[:3])}"
            for gap in gaps[:5]
        )

    def _budget(self, ratio: float) -> int:
        return int(self._config.context.target_tokens * ratio)

    def planner_context(self, state: ResearchState) -> NodeContext:
        return NodeContext(
            permanent=f"Pregunta principal: {state['query']}",
            strategic="Aun no hay subpreguntas. Debes producir agenda inicial.",
            operational=(
                "Descompone la pregunta en subpreguntas concretas, hipotesis de trabajo y consultas de busqueda iniciales."
            ),
        )

    def extractor_context(
        self,
        state: ResearchState,
        *,
        target_subquery_ids: list[str],
        local_source: str,
    ) -> NodeContext:
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=target_subquery_ids,
            budget_tokens=self._budget(self._config.context.evidence_budget_ratio),
        )
        return NodeContext(
            permanent=f"Pregunta principal: {state['query']}",
            strategic=(
                f"Subpreguntas activas:\n{self._render_subqueries(state['active_subqueries'])}\n"
                f"Huecos:\n{self._render_gaps(state['open_gaps'])}"
            ).strip(),
            operational="Extrae evidencia atomica trazable sin inferir mas alla del texto.",
            evidentiary=evidence,
            local_source=local_source,
        )

    def evaluator_context(self, state: ResearchState) -> NodeContext:
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=[item.id for item in state["active_subqueries"]],
            budget_tokens=self._budget(self._config.context.evidence_budget_ratio),
        )
        strategic = (
            f"Subpreguntas activas:\n{self._render_subqueries(state['active_subqueries'])}\n"
            f"Subpreguntas resueltas:\n{self._render_subqueries(state['resolved_subqueries'])}\n"
            f"Huecos:\n{self._render_gaps(state['open_gaps'])}"
        ).strip()
        operational = (
            "Evalua cobertura, contradicciones y suficiencia. No cierres la investigacion si la evidencia no sostiene la respuesta."
        )
        return NodeContext(
            permanent=f"Pregunta principal: {state['query']}",
            strategic=strategic,
            operational=operational,
            evidentiary=evidence,
        )

    def synthesizer_context(self, state: ResearchState) -> NodeContext:
        evidence = select_evidence_for_context(
            state["atomic_evidence"],
            subquery_ids=[item.id for item in state["resolved_subqueries"] or state["active_subqueries"]],
            budget_tokens=self._budget(self._config.context.evidence_budget_ratio),
        )
        budget = self._budget(self._config.context.dossier_budget_ratio)
        subquery_blocks = []
        for subquery in [*state["resolved_subqueries"], *state["active_subqueries"]]:
            summary = state["working_dossier"].subquery_summaries.get(subquery.id, "")
            if not summary:
                continue
            subquery_blocks.append(f"{subquery.id} | {subquery.question}\n{summary}")
        contradictions = "\n".join(
            f"- {item.topic}: {item.statement_a} <> {item.statement_b}"
            for item in state["contradictions"][:4]
        )
        gaps = self._render_gaps(state["open_gaps"])
        joined_subquery_blocks = "\n\n".join(subquery_blocks)
        dossier = "\n\n".join(
            block
            for block in (
                f"Resumen global:\n{state['working_dossier'].global_summary}",
                f"Cobertura por subpregunta:\n{joined_subquery_blocks}" if joined_subquery_blocks else "",
                f"Contradicciones detectadas:\n{contradictions}" if contradictions else "",
                f"Huecos abiertos:\n{gaps}" if gaps else "",
            )
            if block
        )
        if estimate_tokens(dossier) > budget:
            dossier = dossier[: budget * 4]
        return NodeContext(
            permanent=f"Pregunta principal: {state['query']}",
            strategic=dossier,
            operational=(
                "Redacta un informe final en Markdown conceptual con resumen ejecutivo, analisis por temas, confianza, reservas, huecos y citas. "
                "Cada bloque debe apoyarse solo en evidencia relevante."
            ),
            evidentiary=evidence,
        )
