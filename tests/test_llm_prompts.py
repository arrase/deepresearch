from deepresearch.config import ResearchConfig
from deepresearch.prompting import PromptTemplateLoader

def test_load_format_instructions_loads_from_disk() -> None:
    config = ResearchConfig.load()
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
