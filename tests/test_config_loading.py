from __future__ import annotations

import argparse

from deepresearch.config import ResearchConfig
from deepresearch.main import apply_cli_overrides
from deepresearch.prompting import PromptTemplateLoader


def test_load_bootstraps_project_style_config_root(tmp_path) -> None:
    config_root = tmp_path / "project-config"
    config = ResearchConfig.load(config_root=config_root)

    assert config.config_root == config_root.resolve()
    assert config.config_file_path.exists()
    assert config.prompts_dir.exists()
    assert (config.prompts_dir / "planner" / "system.j2").exists()


def test_prompt_loader_renders_user_editable_template(tmp_path) -> None:
    prompts_dir = tmp_path / "prompts"
    planner_dir = prompts_dir / "planner"
    planner_dir.mkdir(parents=True)
    (planner_dir / "system.j2").write_text("System {{ value }}", encoding="utf-8")
    (planner_dir / "human.j2").write_text("Human {{ value }}", encoding="utf-8")

    loader = PromptTemplateLoader(prompts_dir)
    rendered = loader.render("planner", {"value": "template"})

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
        artifacts_dir=str(tmp_path / "artifacts"),
        logs_dir=str(tmp_path / "logs"),
    )

    apply_cli_overrides(config, args)

    assert config.model.model_name == "custom-model"
    assert config.model.num_ctx == 32768
    assert config.context.target_tokens == 32768
    assert config.runtime.max_iterations == 5
    assert config.context.configured_by == "cli_override"