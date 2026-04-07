from deepresearch.config import ResearchConfig
from deepresearch.prompting import PromptTemplateLoader


def test_load_format_instructions_loads_from_disk(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    # Test meta_planner format
    meta_format = loader.render_format("meta_planner", {})
    assert "chapters" in meta_format

    # Test micro_planner format
    micro_format = loader.render_format("micro_planner", {})
    assert "subtopics" in micro_format

    # Test extractor format
    extractor_format = loader.render_format("extractor", {})
    assert "evidences" in extractor_format

    # Test auditor format
    auditor_format = loader.render_format("auditor", {})
    assert "approved" in auditor_format

    # Test sub_synthesizer format
    sub_synth_format = loader.render_format("sub_synthesizer", {})
    assert "chapter_id" in sub_synth_format

    assert '"topic_ids"' in micro_format
    assert '"tags"' not in extractor_format


def test_meta_planner_prompt_mentions_chapters_and_hypotheses(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    rendered = loader.render(
        "meta_planner",
        {
            "query": "What is Tavily?",
            "today_date": "2025-01-01",
            "hypotheses": ["Tavily is a research API"],
            "max_chapters": 5,
            "min_chapters": 3,
            "format_instructions": "",
        },
    )

    assert "chapter" in rendered.human.lower()
    assert "Tavily" in rendered.human
    assert "research API" in rendered.human


def test_micro_planner_prompt_mentions_chapter_context(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    rendered = loader.render(
        "micro_planner",
        {
            "query": "What is Tavily?",
            "today_date": "2025-01-01",
            "chapter_id": "topic_1",
            "chapter_question": "What is Tavily?",
            "chapter_rationale": "Need product definition",
            "chapter_criteria": "- At least one accepted evidence",
            "existing_subtopics": "- None",
            "open_gaps": "- None",
            "dossier_context": "",
            "coverage_summary": "- No evidence yet",
            "format_instructions": "",
        },
    )

    assert "topic_1" in rendered.human or "Tavily" in rendered.human


def test_global_synthesizer_prompt_includes_chapter_drafts(tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")
    loader = PromptTemplateLoader(config.prompts_dir)

    rendered = loader.render(
        "global_synthesizer",
        {
            "query": "What is Tavily?",
            "chapters": [
                {
                    "title": "Capabilities",
                    "executive_summary": "Tavily provides web search results.",
                    "key_findings": ["Good API"],
                    "sections": [],
                    "limitations": [],
                    "open_gaps": [],
                    "confidence": "high",
                }
            ],
            "global_limitations": [],
            "language": "English",
            "format_instructions": "",
        },
    )

    assert "Capabilities" in rendered.human
    assert "Tavily" in rendered.human
