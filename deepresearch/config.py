"""Configuracion tipada del sistema de investigacion.

La configuracion centraliza el presupuesto de contexto, parametros del modelo y
politicas operativas. El objetivo es que los nodos no introduzcan presupuestos
propios de forma ad hoc: cada etapa consulta esta configuracion y adapta su
seleccion de contexto al tamano declarado por el usuario.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Configuracion de ChatOllama optimizada para modelos pequenos.

    La ventana se expone como parametro de usuario. El valor por defecto sigue
    el requisito operativo del proyecto: qwen3.5:9b con 100000 tokens de
    contexto objetivo.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(default="qwen3.5:9b")
    base_url: str = Field(default="http://127.0.0.1:11434")
    temperature_planner: float = Field(default=0.2, ge=0.0, le=1.0)
    temperature_extractor: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature_evaluator: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature_synthesizer: float = Field(default=0.1, ge=0.0, le=1.0)
    num_ctx: int = Field(default=100000, ge=4096)
    num_predict: int = Field(default=900, ge=64)
    timeout_seconds: int = Field(default=120, ge=5)


class ContextPolicyConfig(BaseModel):
    """Presupuesto global del contexto y reglas de seleccion."""

    model_config = ConfigDict(extra="forbid")

    target_tokens: int = Field(default=100000, ge=4096)
    configured_by: str = Field(default="cli")
    selection_policy: str = Field(default="hierarchical_relevance_first")
    evidence_budget_ratio: float = Field(default=0.45, gt=0.0, lt=1.0)
    dossier_budget_ratio: float = Field(default=0.30, gt=0.0, lt=1.0)
    local_source_budget_ratio: float = Field(default=0.20, gt=0.0, lt=1.0)
    safety_margin_ratio: float = Field(default=0.05, ge=0.0, lt=0.5)


class BrowserConfig(BaseModel):
    """Configuracion del runtime de Lightpanda gestionado por Docker."""

    model_config = ConfigDict(extra="forbid")

    image: str = Field(default="lightpanda/browser:nightly")
    disable_telemetry: bool = Field(default=True)
    wait_ms: int = Field(default=7000, ge=250)
    wait_until: str = Field(default="networkidle")
    obey_robots: bool = Field(default=True)
    max_content_chars: int = Field(default=24000, ge=1000)
    min_useful_chars: int = Field(default=300, ge=50)
    min_partial_chars: int = Field(default=120, ge=20)
    request_timeout_seconds: int = Field(default=90, ge=5)


class SearchConfig(BaseModel):
    """Configuracion del descubrimiento inicial de fuentes."""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(default="duckduckgo_html")
    results_per_query: int = Field(default=5, ge=1, le=20)
    max_queries_per_cycle: int = Field(default=3, ge=1, le=10)
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    )


class RuntimeConfig(BaseModel):
    """Limites tecnicos y rutas de artefactos.

    Los limites existen como salvaguarda, no como criterio principal de parada.
    """

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=8, ge=1)
    artifacts_dir: Path = Field(default=Path("artifacts"))
    logs_dir: Path = Field(default=Path("logs"))
    enable_checkpoints: bool = Field(default=True)
    llm_retry_attempts: int = Field(default=2, ge=0, le=5)


class ResearchConfig(BaseModel):
    """Configuracion raiz consumida por el runtime del grafo."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextPolicyConfig = Field(default_factory=ContextPolicyConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    def ensure_directories(self) -> None:
        self.runtime.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.runtime.logs_dir.mkdir(parents=True, exist_ok=True)
