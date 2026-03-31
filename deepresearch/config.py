"""Typed application configuration and bootstrap helpers.

The configuration layer owns model parameters, context policy, prompt asset
discovery, and runtime limits. Nodes consume these validated settings instead
of inventing stage-specific defaults at call sites.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tomllib
import sys
from importlib import resources

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
    """Global context budget and deterministic selection policy."""

    model_config = ConfigDict(extra="forbid")

    evidence_budget_ratio: float = Field(default=0.45, gt=0.0, lt=1.0)
    dossier_budget_ratio: float = Field(default=0.30, gt=0.0, lt=1.0)
    local_source_budget_ratio: float = Field(default=0.20, gt=0.0, lt=1.0)
    safety_margin_ratio: float = Field(default=0.05, ge=0.0, lt=0.5)


class BrowserConfig(BaseModel):
    """Runtime settings for the Docker-managed Lightpanda browser."""

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
    """Search backend and discovery settings."""

    model_config = ConfigDict(extra="forbid")

    backend: str = Field(default="duckduckgo_lite")
    api_key: str | None = Field(default=None)
    results_per_query: int = Field(default=5, ge=1, le=20)
    max_queries_per_cycle: int = Field(default=3, ge=1, le=10)
    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        )
    )


class DiscordConfig(BaseModel):
    """Discord notification settings."""

    model_config = ConfigDict(extra="forbid")

    token: str | None = Field(default=None)
    user_id: str | None = Field(default=None)
    output: str = Field(default="pdf", pattern="^(markdown|pdf)$")


class RuntimeConfig(BaseModel):
    """Technical safeguards and output locations."""

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=8, ge=1)
    llm_retry_attempts: int = Field(default=2, ge=0, le=5)
    language: str = Field(default="English")


class ResearchConfig(BaseModel):
    """Root configuration consumed by the research runtime."""

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    context: ContextPolicyConfig = Field(default_factory=ContextPolicyConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    discord: DiscordConfig = Field(default_factory=DiscordConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    _config_root: Path = PrivateAttr(default_factory=default_config_root)
    _config_file_path: Path = PrivateAttr(default_factory=lambda: default_config_root() / DEFAULT_CONFIG_FILENAME)

    @classmethod
    def load(cls, *, config_root: str | Path | None = None) -> ResearchConfig:
        resolved_root = resolve_config_root(config_root)
        config_file_path = resolved_root / DEFAULT_CONFIG_FILENAME
        
        if not config_file_path.exists():
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
