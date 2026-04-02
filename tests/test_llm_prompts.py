from deepresearch.config import ResearchConfig
from deepresearch.prompting import PromptTemplateLoader


def test_load_format_instructions_loads_from_disk(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    # Test planner format
    planner_format = loader.render_format("planner", {})
    assert "subqueries" in planner_format

    # Test extractor format
    extractor_format = loader.render_format("extractor", {})
    assert "evidences" in extractor_format

    # Test evaluator format
    evaluator_format = loader.render_format("evaluator", {})
    assert "resolved_subquery_ids" in evaluator_format


def test_planner_prompt_mentions_answer_shape_and_coverage_snapshots(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    rendered = loader.render(
        "planner",
        {
            "query": "What is Lightpanda?",
            "has_subqueries": True,
            "dossier_context": "Some dossier\n",
            "coverage_summary": "- sq_1 ...",
            "source_balance_summary": "- lightpanda.io (3)",
            "active_subqueries": "- sq_1: capabilities",
            "resolved_subqueries": "- None",
            "open_gaps": "- sq_1: need limitations",
            "format_instructions": "",
        },
    )

    assert "Coverage Snapshot" in rendered.human
    assert "Source Balance Snapshot" in rendered.human
    assert "infer what kind of answer the user is asking for" in rendered.human


def test_evaluator_prompt_mentions_structural_coverage(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    rendered = loader.render(
        "evaluator",
        {
            "query": "What is Lightpanda?",
            "coverage_summary": "- sq_1 ...",
            "source_balance_summary": "- lightpanda.io (3)",
            "dossier_context": "Some dossier\n",
            "active_subqueries": "- sq_1: capabilities",
            "resolved_subqueries": "- None",
            "open_gaps": "- sq_1: need limitations",
            "evidentiary": "- sq_1: claim | source=Vendor docs | domain=vendor.example",
            "format_instructions": "",
        },
    )

    assert "structural coverage" in rendered.system
    assert "answer shape demanded by the user question" in rendered.human
