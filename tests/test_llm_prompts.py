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
    assert '"topic_ids"' in planner_format
    assert '"topic_id"' in evaluator_format
    assert '"tags"' not in extractor_format


def test_planner_prompt_mentions_answer_shape_and_coverage_snapshots(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    rendered = loader.render(
        "planner",
        {
            "query": "What is Tavily?",
            "has_subqueries": True,
            "dossier_context": "Some dossier\n",
            "coverage_summary": "- topic_1 ...",
            "source_balance_summary": "- tavily.com (3)",
            "active_subqueries": "- topic_1: capabilities",
            "resolved_subqueries": "- None",
            "open_gaps": "- topic_1: need limitations",
            "format_instructions": "",
        },
    )

    assert "Coverage summary" in rendered.human
    assert "Source balance" in rendered.human
    assert "search_intents must point to topic ids through topic_ids" in rendered.human


def test_evaluator_prompt_mentions_structural_coverage(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    rendered = loader.render(
        "evaluator",
        {
            "query": "What is Tavily?",
            "coverage_summary": "- topic_1 ...",
            "source_balance_summary": "- tavily.com (3)",
            "dossier_context": "Some dossier\n",
            "active_subqueries": "- topic_1: capabilities",
            "resolved_subqueries": "- None",
            "open_gaps": "- topic_1: need limitations",
            "evidentiary": "- topic_1: claim | source=Vendor docs | domain=vendor.example",
            "language": "English",
            "format_instructions": "",
        },
    )

    assert "small-model research pipeline" in rendered.system
    assert "resolved_subquery_ids" in rendered.human


def test_synthesizer_prompt_avoids_fake_inline_citations(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    rendered = loader.render(
        "synthesizer",
        {
            "query": "What is Tavily?",
            "coverage_summary": "- topic_1 [completed] ...",
            "source_balance_summary": "- Unique evidence domains: 2",
            "open_gaps": "- None",
            "dossier_context": "topic_1 | What is Tavily?\nA web research API.",
            "topic_briefs_context": "## What is Tavily?\n### Answer\nIt is a web research API.",
            "evidentiary": "- evidence_1 | topic=topic_1 | claim=Tavily provides web search results with raw content.",
            "language": "English",
            "target_words": 1800,
            "format_instructions": "",
        },
    )

    assert "Do not fabricate inline citations" in rendered.human
    assert "runtime will append the final referenced-sources section separately" in rendered.human
    assert "Topic Briefs" in rendered.human
