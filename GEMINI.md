# Deep Research: Project Context

Deep Research is an auditable research pipeline designed to transform open-ended questions into bounded, high-quality research reports with explicit citations. It utilizes a stateful graph-based architecture to manage a research loop involving planning, source discovery, web browsing, evidence extraction, and synthesis.

## Core Architecture & Technologies

- **Workflow Orchestration**: Built with [LangGraph](https://github.com/langchain-ai/langgraph), using an explicit `ResearchState` to manage the research lifecycle without relying on conversation history.
- **LLM Backend**: Powered by [Ollama](https://ollama.com/) (ChatOllama) for all reasoning, extraction, and synthesis tasks.
- **Browser Backend**: Uses [Lightpanda](https://lightpanda.io/) (via Docker) as a headless browser for reliable web scraping.
- **Search Backend**: Default search is via DuckDuckGo Lite HTML parsing.
- **Prompt Management**: Prompts are stored as [Jinja2](https://jinja.palletsprojects.com/) templates in the `config/prompts/` directory, allowing for easy customization without modifying Python code.
- **Configuration**: Uses TOML (`config/config.toml`) for managing model settings, context window policies, and runtime limits.

## Project Structure

- `deepresearch/`: Core Python package.
  - `graph.py`: Defines the LangGraph `StateGraph` and routing logic.
  - `nodes.py`: Contains the `ResearchNodes` class, implementing the logic for each graph step (planner, browser, extractor, etc.).
  - `state.py`: Defines the `ResearchState` TypedDict and other data models (Pydantic).
  - `subagents/`: Specialized workers for LLM tasks and deterministic data processing.
  - `context_manager.py`: Manages the assembly of context for LLM prompts.
  - `telemetry.py`: Handles logging, artifact generation, and checkpoints.
- `config/`: Configuration root.
  - `config.toml`: Main project configuration.
  - `prompts/`: Jinja2 templates for each research stage (planner, extractor, evaluator, synthesizer, repair).
- `artifacts/`: Default directory for generated reports and session checkpoints.
- `tests/`: Comprehensive test suite using `pytest`.

## Building and Running

### Prerequisites
- Python 3.11 or higher.
- [Ollama](https://ollama.com/) installed and running.
- [Docker](https://www.docker.com/) installed and running (for Lightpanda).

### Installation
```bash
pip install -e .
```

### Running Research
```bash
# Run a basic research session
deepresearch run "What are the latest developments in fusion energy?"

# Run with a custom configuration root
deepresearch run "..." --config-root ./my-config

# Validate infrastructure (Docker, Ollama, Lightpanda)
deepresearch self-check
```

### Running Tests
```bash
pytest
```

## Development Conventions

- **State over History**: Never rely on LLM conversation history for state. All relevant information must be persisted in the `ResearchState`.
- **Prompt Decoupling**: Do NOT use inline strings for LLM prompts. Always use the Jinja2 templates in `config/prompts/`.
- **Deterministic Processing**: Use deterministic workers (in `deepresearch/subagents/deterministic.py`) for data cleanup, scoring, and deduplication to ensure consistency.
- **Type Safety**: Use Pydantic models for data structures and maintain strict type hinting throughout the codebase.
- **Telemetry & Artifacts**: All significant events should be recorded via the `TelemetryRecorder`. Every run should produce a markdown report and a final state checkpoint in the `artifacts/` directory.
- **Browser Usage**: Web interaction is handled through `LightpandaDockerManager`. Ensure the Docker daemon is accessible during runtime.
