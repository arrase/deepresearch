"""Typed application configuration and bootstrap helpers.

The configuration layer owns model parameters, context policy, prompt asset
discovery, and runtime limits. Nodes consume these validated settings instead
of inventing stage-specific defaults at call sites.
"""

from __future__ import annotations

import tomllib
from importlib import resources
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

DEFAULT_CONFIG_ENV_VAR = "DEEPRESEARCH_CONFIG_ROOT"
DEFAULT_CONFIG_FILENAME = "config.toml"


def default_config_root() -> Path:
    """Returns the source config directory in the project root or package resources."""
    # Try to get the path from resources (works in installed environments)
    try:
        return Path(str(resources.files("deepresearch.resources")))
    except (ImportError, TypeError):
        # Fallback for development if resources are not yet available as a package
        return Path(__file__).resolve().parent.parent / "config"


def resolve_config_root(override: str | Path | None = None) -> Path:
    """
    Resolve the config root directory.
    Priority:
    1. Explicit override
    2. ~/.deepresearch/config
    """
    if override is not None:
        return Path(override).expanduser().resolve()

    return (Path.home() / ".deepresearch" / "config").resolve()


def bootstrap_config_root(config_root: Path) -> None:
    """Materialize an editable config tree from the project defaults when needed."""

    # Get the source root as a Traversable object
    source_root = resources.files("deepresearch.resources")

    config_root = config_root.resolve()
    config_root.mkdir(parents=True, exist_ok=True)

    # We can't easily compare Traversable to Path for equality if installed as wheel,
    # but we only bootstrap if config_root doesn't exist or is missing files.

    _copy_resource_to_path(source_root / DEFAULT_CONFIG_FILENAME, config_root / DEFAULT_CONFIG_FILENAME)
    _copy_resource_tree(source_root / "prompts", config_root / "prompts")


def _copy_resource_tree(source: resources.abc.Traversable, target: Path) -> None:
    """Recursively copy a Traversable resource tree to a physical Path."""
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_resource_tree(child, target / child.name)
        return

    # It's a file
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _copy_resource_to_path(source: resources.abc.Traversable, target: Path) -> None:
    """Copy a single resource file to a physical Path if it doesn't exist."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())


class ModelConfig(BaseModel):
    """ChatOllama settings tuned for smaller local models."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        default="qwen3.5:9b",
        description="Ollama model name used for all research stages.",
    )
    base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL of the local or remote Ollama server.",
    )
    temperature_planner: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for the planning stage.",
    )
    temperature_extractor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for evidence extraction.",
    )
    temperature_evaluator: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for coverage evaluation.",
    )
    temperature_synthesizer: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for final report synthesis.",
    )
    temperature_topic_synthesizer: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for topic-brief synthesis.",
    )
    num_ctx: int = Field(default=100000, ge=4096, description="Maximum context window passed to Ollama.")
    num_predict: int = Field(default=8192, ge=64, description="Maximum number of tokens generated per LLM call.")
    timeout_seconds: int = Field(default=120, ge=5, description="Per-request timeout for Ollama calls.")


class SearchConfig(BaseModel):
    """Search backend and discovery settings."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = Field(default=None, description="Tavily API key used for web search requests.")
    results_per_query: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum Tavily results requested for each search query.",
    )
    max_raw_content_chars: int = Field(
        default=24000,
        ge=1000,
        description="Maximum raw page characters kept from each search result.",
    )
    min_source_chars: int = Field(
        default=300,
        ge=50,
        description="Minimum source content length required before extraction.",
    )


class ReporterConfig(BaseModel):
    """Public controls for the final synthesis budget."""

    model_config = ConfigDict(extra="forbid")

    output_reserve_ratio: float = Field(
        default=0.20,
        ge=0.05,
        lt=0.8,
        description="Fraction of the context window reserved for the final answer.",
    )
    prompt_margin_tokens: int = Field(
        default=512,
        ge=0,
        description="Extra prompt headroom kept free before synthesis.",
    )
    topic_brief_budget_ratio: float = Field(
        default=0.45,
        ge=0.1,
        le=0.8,
        description="Share of the synthesis prompt budget reserved for topic briefs.",
    )
    final_report_target_words: int = Field(
        default=1800,
        ge=400,
        le=5000,
        description="Approximate target length for rich final reports.",
    )


class DedupConfig(BaseModel):
    """Public controls for conservative lexical deduplication."""

    model_config = ConfigDict(extra="forbid")

    lexical_fingerprint: bool = Field(
        default=True,
        description="Enable lexical fingerprint checks before keeping near-duplicate evidence.",
    )
    approximate_jaccard_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Similarity threshold used to reject near-duplicate evidence.",
    )
    min_length_ratio: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Lower length ratio bound when comparing duplicate candidates.",
    )
    max_length_ratio: float = Field(
        default=1.25,
        ge=1.0,
        le=5.0,
        description="Upper length ratio bound when comparing duplicate candidates.",
    )


class DiscordConfig(BaseModel):
    """Discord notification settings."""

    model_config = ConfigDict(extra="forbid")

    token: str | None = Field(default=None, description="Discord bot token used to send direct messages.")
    user_id: str | None = Field(default=None, description="Discord user ID that receives the report.")
    output: str = Field(
        default="pdf",
        pattern="^(markdown|pdf)$",
        description="Attachment format used when the report is too long for a message.",
    )


class RuntimeConfig(BaseModel):
    """Technical safeguards and output locations."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=8, ge=1, description="Hard cap on planner and search cycles for a run.")
    search_batch_size: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many candidate search queries to execute per cycle.",
    )
    min_attempts_before_exhaustion: int = Field(
        default=3,
        ge=1,
        description="Minimum attempts before the runtime can mark a topic as exhausted.",
    )
    max_cycles_without_new_evidence: int = Field(
        default=4,
        ge=1,
        description="Stop after this many cycles without newly accepted evidence.",
    )
    max_cycles_without_useful_sources: int = Field(
        default=4,
        ge=1,
        description="Stop after this many cycles without useful sources.",
    )
    max_consecutive_technical_failures: int = Field(
        default=3,
        ge=1,
        description="Abort when too many consecutive technical failures happen.",
    )
    semantic_eval_interval: int = Field(
        default=0,
        ge=0,
        description="Run evaluator every N cycles even without strong evidence updates; 0 disables it.",
    )
    allow_dynamic_replan: bool = Field(
        default=True,
        description="Allow the planner to revise the topic plan during the run.",
    )
    verbosity: int = Field(default=0, ge=0, le=3, description="CLI log verbosity from quiet to detailed diagnostics.")
    llm_retry_attempts: int = Field(
        default=2,
        ge=0,
        le=5,
        description="How many times to retry recoverable LLM parsing failures.",
    )
    min_topic_evidence_target: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Minimum evidence target applied after planner normalization.",
    )
    max_topic_evidence_target: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Upper bound for dynamically normalized topic evidence targets.",
    )
    extraction_max_chars_per_pass: int = Field(
        default=4000,
        ge=1000,
        description="Maximum source characters sent to one extraction pass.",
    )
    max_extraction_passes_per_source: int = Field(
        default=3,
        ge=1,
        le=6,
        description="How many extraction passes can be run for one source.",
    )
    language: str = Field(default="English", description="Language used for prompts and the final report.")


class LangSmithConfig(BaseModel):
    """LangSmith tracing settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Enable LangSmith integration for this run.")
    tracing: bool = Field(default=True, description="Emit tracing spans when LangSmith integration is enabled.")
    endpoint: str | None = Field(
        default=None,
        description="Custom LangSmith API endpoint, if you are not using the default service.",
    )
    api_key: str | None = Field(default=None, description="LangSmith API key required when tracing is enabled.")
    project: str = Field(default="DeepResearch", description="LangSmith project name used for uploaded traces.")

    @model_validator(mode="after")
    def validate_enabled_credentials(self) -> LangSmithConfig:
        if self.enabled and not self.api_key:
            raise ValueError("langsmith.api_key is required when langsmith.enabled is true")
        return self


class ResearchConfig(BaseModel):
    """Root configuration consumed by the research runtime."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="LLM settings shared by the research pipeline.",
    )
    search: SearchConfig = Field(
        default_factory=SearchConfig,
        description="Search backend credentials and retrieval limits.",
    )
    reporter: ReporterConfig = Field(
        default_factory=ReporterConfig,
        description="Budget settings for the final synthesis prompt.",
    )
    dedup: DedupConfig = Field(
        default_factory=DedupConfig,
        description="Deduplication rules for evidence selection.",
    )
    discord: DiscordConfig = Field(
        default_factory=DiscordConfig,
        description="Optional Discord delivery settings.",
    )
    runtime: RuntimeConfig = Field(
        default_factory=RuntimeConfig,
        description="Execution limits, retry policy, and logging verbosity.",
    )
    langsmith: LangSmithConfig = Field(
        default_factory=LangSmithConfig,
        description="Optional tracing integration settings.",
    )

    _config_root: Path = PrivateAttr(default_factory=default_config_root)
    _config_file_path: Path = PrivateAttr(default_factory=lambda: default_config_root() / DEFAULT_CONFIG_FILENAME)

    @classmethod
    def load(cls, *, config_root: str | Path | None = None) -> ResearchConfig:
        resolved_root = resolve_config_root(config_root)
        config_file_path = resolved_root / DEFAULT_CONFIG_FILENAME

        bootstrap_config_root(resolved_root)

        raw_payload = tomllib.loads(config_file_path.read_text(encoding="utf-8"))
        config = cls.model_validate(raw_payload)
        config._config_root = resolved_root
        config._config_file_path = config_file_path
        return config

    @property
    def config_root(self) -> Path:
        return self._config_root

    @property
    def config_file_path(self) -> Path:
        return self._config_file_path

    @property
    def prompts_dir(self) -> Path:
        return self._config_root / "prompts"
