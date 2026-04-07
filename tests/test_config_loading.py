from __future__ import annotations

import argparse
import sys

import pytest
from pydantic import ValidationError

from deepresearch.config import ResearchConfig
from deepresearch.main import apply_cli_overrides, parse_args
from deepresearch.prompting import PromptTemplateLoader


def test_load_bootstraps_project_style_config_root(tmp_path) -> None:
    config_root = tmp_path / "project-config"
    config = ResearchConfig.load(config_root=config_root)

    assert config.config_root == config_root.resolve()
    assert config.config_file_path.exists()
    assert config.prompts_dir.exists()
    assert (config.prompts_dir / "meta_planner" / "system.j2").exists()


def test_bootstrap_config_file_contains_inline_help_comments(tmp_path) -> None:
    config_root = tmp_path / "project-config"
    config = ResearchConfig.load(config_root=config_root)
    config_text = config.config_file_path.read_text(encoding="utf-8")

    assert 'api_key = "" # Tavily API key used for web search requests.' in config_text
    assert 'max_iterations = 8 # Hard cap on total search cycles across all chapters.' in config_text


def test_prompt_loader_renders_user_editable_template(tmp_path) -> None:
    prompts_dir = tmp_path / "prompts"
    meta_planner_dir = prompts_dir / "meta_planner"
    meta_planner_dir.mkdir(parents=True)
    (meta_planner_dir / "system.j2").write_text("System {{ value }}", encoding="utf-8")
    (meta_planner_dir / "human.j2").write_text("Human {{ value }}", encoding="utf-8")

    loader = PromptTemplateLoader(prompts_dir)
    rendered = loader.render("meta_planner", {"value": "template"})

    assert rendered.system == "System template"
    assert rendered.human == "Human template"


def test_cli_overrides_take_precedence_over_toml_config(tmp_path) -> None:
    config_root = tmp_path / "config-root"
    config = ResearchConfig.load(config_root=config_root)
    args = argparse.Namespace(
        config_root=str(config_root),
        model="custom-model",
        num_ctx=32768,
        max_iterations=5,
        verbosity=3,
    )

    apply_cli_overrides(config, args)

    assert config.model.model_name == "custom-model"
    assert config.model.num_ctx == 32768
    assert config.runtime.max_iterations == 5
    assert config.runtime.verbosity == 3


def test_loaded_config_uses_runtime_synthesis_budget_settings(tmp_path) -> None:
    config_root = tmp_path / "config-root"
    config = ResearchConfig.load(config_root=config_root)

    assert config.reporter.output_reserve_ratio == 0.20
    assert config.reporter.prompt_margin_tokens == 512
    assert config.runtime.max_consecutive_technical_failures == 3
    assert config.runtime.max_cycles_without_new_evidence == 4
    assert config.runtime.max_cycles_without_useful_sources == 4
    assert config.runtime.search_batch_size == 3
    assert config.runtime.min_attempts_before_exhaustion == 3
    assert config.runtime.max_chapters == 5
    assert config.runtime.max_topic_depth == 2
    assert config.runtime.max_audit_rejections == 2


def test_load_rejects_unknown_root_sections(tmp_path) -> None:
    config_root = tmp_path / "invalid-config-root"
    config = ResearchConfig.load(config_root=config_root)
    config_text = config.config_file_path.read_text(encoding="utf-8")
    config.config_file_path.write_text(
        config_text.replace(
            "[search]",
            "[context]\n"
            "evidence_budget_ratio = 0.45\n"
            "dossier_budget_ratio = 0.30\n"
            "local_source_budget_ratio = 0.20\n"
            "safety_margin_ratio = 0.05\n\n"
            "[search]",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        ResearchConfig.load(config_root=config_root)


def test_runtime_progress_thresholds_are_user_editable(tmp_path) -> None:
    config_root = tmp_path / "custom-runtime-config"
    config = ResearchConfig.load(config_root=config_root)
    config_text = config.config_file_path.read_text(encoding="utf-8")
    config.config_file_path.write_text(
        config_text.replace("max_cycles_without_new_evidence = 4", "max_cycles_without_new_evidence = 7")
        .replace("max_cycles_without_useful_sources = 4", "max_cycles_without_useful_sources = 5")
        .replace("max_consecutive_technical_failures = 3", "max_consecutive_technical_failures = 5")
        .replace("search_batch_size = 3", "search_batch_size = 2"),
        encoding="utf-8",
    )

    reloaded = ResearchConfig.load(config_root=config_root)

    assert reloaded.runtime.max_cycles_without_new_evidence == 7
    assert reloaded.runtime.max_cycles_without_useful_sources == 5
    assert reloaded.runtime.max_consecutive_technical_failures == 5
    assert reloaded.runtime.search_batch_size == 2


def test_load_rejects_unknown_runtime_fields(tmp_path) -> None:
    config_root = tmp_path / "invalid-runtime-config"
    config = ResearchConfig.load(config_root=config_root)
    config_text = config.config_file_path.read_text(encoding="utf-8")
    config.config_file_path.write_text(
        config_text.replace(
            'language = "English"',
            'language = "English"\nunknown_runtime_field = 99',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        ResearchConfig.load(config_root=config_root)


def test_runtime_default_verbosity_is_zero(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")

    assert config.runtime.verbosity == 0


def test_parse_args_accepts_verbosity(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["deepresearch", "What is Tavily search?", "--verbosity", "3"])

    args = parse_args()

    assert args.query == "What is Tavily search?"
    assert args.verbosity == 3
