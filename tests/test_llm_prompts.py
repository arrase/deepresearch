from deepresearch.config import ResearchConfig
from deepresearch.subagents.llm import LLMWorkers

def test_load_format_instructions_loads_from_disk() -> None:
    config = ResearchConfig.load()
    workers = LLMWorkers(config)
    
    # Test planner format
    planner_format = workers._load_format_instructions("planner", {})
    assert "subqueries" in planner_format
    assert "search_intents" in planner_format
    
    # Test extractor format
    extractor_format = workers._load_format_instructions("extractor", {})
    assert "evidences" in extractor_format
    
    # Test evaluator format
    evaluator_format = workers._load_format_instructions("evaluator", {})
    assert "resolved_subquery_ids" in evaluator_format
    
    # Test synthesizer format
    synthesizer_format = workers._load_format_instructions("synthesizer", {})
    # Format for synthesizer is now empty to bypass JSON parsing

def test_load_format_instructions_fallback_on_missing_file() -> None:
    config = ResearchConfig.load()
    workers = LLMWorkers(config)
    
    # Test fallback for non-existent prompt
    fallback = workers._load_format_instructions("non_existent_prompt", {})
    assert fallback == "Return valid JSON only, with no additional text."
