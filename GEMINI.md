# Deep Research: Project Context

Deep Research is an auditable research pipeline designed to transform open-ended questions into bounded, high-quality research reports with explicit citations. It utilizes a stateful graph-based architecture to manage a research loop involving planning, source discovery, web browsing, evidence extraction, and synthesis.

## Core Architecture & Technologies

- **Workflow Orchestration**: Built with [LangGraph](https://github.com/langchain-ai/langgraph), using an explicit `ResearchState` to manage the research lifecycle without relying on conversation history.
- **LLM Backend**: Powered by [Ollama](https://ollama.com/) (ChatOllama) for all reasoning, extraction, and synthesis tasks.
- **Browser Backend**: Uses [Lightpanda](https://lightpanda.io/) (via Docker) as a high-performance headless browser for reliable web scraping.
- **Search Backend**: Supports DuckDuckGo (default) and Tavily.
- **Prompt Management**: Prompts are stored as [Jinja2](https://jinja.palletsprojects.com/) templates in `config/prompts/`, allowing for easy customization without modifying Python code.
- **Configuration**: Uses TOML (`config/config.toml`) for managing model settings, context window policies, and runtime limits.

## Project Structure

- `deepresearch/`: Core Python package.
  - `graph.py`: Defines the LangGraph `StateGraph` and routing logic.
  - `nodes.py`: Contains the `ResearchNodes` class, implementing the logic for each graph step (planner, browser, extractor, etc.).
  - `state.py`: Defines the `ResearchState` TypedDict and other data models (Pydantic).
  - `subagents/`: Specialized workers for LLM tasks and deterministic data processing.
  - `context_manager.py`: Manages the assembly of context for LLM prompts.
  - `telemetry.py`: Handles console logging.
  - `tools.py`: Search and browser clients.
- `config/`: Configuration root.
  - `config.toml`: Global settings.
  - `prompts/`: Jinja2 templates for different stages (planner, extractor, synthesizer, etc.).
- `tests/`: Comprehensive test suite for state, tools, and end-to-end graph execution.

## Research Workflow

The research process follows a directed graph:

1.  **Planner**: Generates a research agenda with subqueries and search intents.
2.  **Source Manager**: Discovers and prioritizes candidate URLs.
3.  **Browser**: Fetches and cleans content using Lightpanda.
4.  **Extractor**: Pulls atomic evidence (claims, quotations, citations).
5.  **Context Manager**: Integrates evidence into the "Working Dossier".
6.  **Evaluator**: Assesses research coverage and decides if more iterations are needed.
7.  **Synthesizer**: Generates the final Markdown report with referenced sources.

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
deepresearch "What are the latest developments in fusion energy?"

# Specify an output file
deepresearch "..." -o my_report.md

# Run with verbose telemetry
deepresearch "..." -v
```

### Running Tests
```bash
pytest
```

## Development Conventions

- **State over History**: Never rely on LLM conversation history for state. All relevant information must be persisted in the `ResearchState`.
- **Prompt Decoupling**: Do NOT use inline strings for LLM prompts. Always use the Jinja2 templates in `config/prompts/`.
- **Deterministic Processing**: Use deterministic workers (in `deepresearch/subagents/deterministic.py`) for data cleanup, scoring, and deduplication.
- **Type Safety**: Use Pydantic models for data structures and maintain strict type hinting.
- **Telemetry**: Significant events are recorded via the `TelemetryRecorder` to the console.
- **Browser Usage**: Web interaction is handled through `LightpandaDockerManager`. Ensure the Docker daemon is accessible.
