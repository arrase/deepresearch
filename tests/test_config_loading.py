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
    )

    apply_cli_overrides(config, args)

    assert config.model.model_name == "custom-model"
    assert config.model.num_ctx == 32768
    assert config.runtime.max_iterations == 5


def test_loaded_config_uses_runtime_synthesis_budget_settings(tmp_path) -> None:
    config_root = tmp_path / "config-root"
    config = ResearchConfig.load(config_root=config_root)

    assert config.runtime.synthesizer_output_reserve_ratio == 0.20
    assert config.runtime.synthesizer_prompt_margin == 512
    assert config.runtime.max_stagnation_cycles == 4
    assert config.runtime.max_consecutive_technical_failures == 3
    assert config.runtime.max_cycles_without_new_evidence == 3
    assert config.runtime.max_cycles_without_useful_sources == 4
    assert config.runtime.min_progress_score_to_reset_stagnation == 2


def test_load_ignores_legacy_context_section(tmp_path) -> None:
    config_root = tmp_path / "legacy-config-root"
    config = ResearchConfig.load(config_root=config_root)
    legacy_text = config.config_file_path.read_text(encoding="utf-8")
    config.config_file_path.write_text(
        legacy_text.replace(
            "[browser]",
            "[context]\n"
            "evidence_budget_ratio = 0.45\n"
            "dossier_budget_ratio = 0.30\n"
            "local_source_budget_ratio = 0.20\n"
            "safety_margin_ratio = 0.05\n\n"
            "[browser]",
        ),
        encoding="utf-8",
    )

    reloaded = ResearchConfig.load(config_root=config_root)

    assert reloaded.runtime.synthesizer_output_reserve_ratio == 0.20


def test_runtime_progress_thresholds_are_user_editable(tmp_path) -> None:
    config_root = tmp_path / "custom-runtime-config"
    config = ResearchConfig.load(config_root=config_root)
    config_text = config.config_file_path.read_text(encoding="utf-8")
    config.config_file_path.write_text(
        config_text.replace("max_stagnation_cycles = 4", "max_stagnation_cycles = 7")
        .replace("max_consecutive_technical_failures = 3", "max_consecutive_technical_failures = 5")
        .replace("weight_actionable_gap = 1", "weight_actionable_gap = 2"),
        encoding="utf-8",
    )

    reloaded = ResearchConfig.load(config_root=config_root)

    assert reloaded.runtime.weight_actionable_gap == 2
    assert reloaded.runtime.max_stagnation_cycles == 7
    assert reloaded.runtime.max_consecutive_technical_failures == 5
