# Deep Research Project Overview

`deepresearch` is an auditable deep research pipeline that leverages LangGraph, Ollama, and Lightpanda to perform comprehensive, multi-step research on open-ended queries. It is designed to be transparent, traceable, and capable of running entirely locally using local LLMs.

## Core Technologies
- **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) for stateful, multi-actor applications.
- **LLM:** [Ollama](https://ollama.com/) for local model execution (e.g., Qwen 2.5).
- **Browsing:** [Lightpanda](https://lightpanda.io/) (via Docker) for high-performance, scriptable web browsing.
- **Search:** [Tavily](https://tavily.com/) or DuckDuckGo for web discovery.
- **Reporting:** Jinja2 for prompt templating and WeasyPrint for PDF generation.

## Architecture
The system operates as a `StateGraph` with the following key nodes:

1.  **Planner:** Analyzes the initial query and breaks it down into subqueries, search intents, and hypotheses.
2.  **Source Manager:** Prioritizes search queries and manages a queue of candidate URLs.
3.  **Browser:** Navigates to selected URLs using a Docker-managed Lightpanda instance.
4.  **Extractor:** Uses the LLM to extract atomic evidence from the page content, mapped back to specific subqueries.
5.  **Context Manager:** Integrates new evidence into a working dossier, performing deduplication and summarization.
6.  **Evaluator:** Assesses research coverage, identifies contradictions or gaps, and decides whether to continue or synthesize the final report.
7.  **Synthesizer:** Generates the final executive answer and structured report with citations.

## Project Structure
- `deepresearch/`: Core source code.
    - `graph.py`: StateGraph assembly and routing logic.
    - `nodes.py`: Implementation of research nodes (Planner, Browser, etc.).
    - `state.py`: Typed state and evidence models using Pydantic.
    - `config.py`: Configuration management (Pydantic settings).
    - `tools.py`: Implementation of search and browser clients.
    - `subagents/`: Specialized LLM and communication workers.
- `config/`: Default configuration and prompt templates.
- `prompts/`: Jinja2 templates for each research stage.
- `tests/`: Comprehensive test suite using `pytest`.

## Development Commands

### Environment Setup
The project requires Python 3.11+ and a running Docker daemon (for Lightpanda).
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Ensure Lightpanda image is available
docker pull lightpanda/browser:nightly
```

### Running Research
```bash
# Run a research query (defaults to markdown output)
deepresearch "How does Lightpanda compare to Playwright for LLM scraping?"

# Save as PDF
deepresearch "The impact of Llama 3 on local AI" --pdf report.pdf

# Verbose mode for real-time telemetry
deepresearch "Quantum computing breakthroughs 2024" -v
```

### Testing
```bash
# Run all tests
pytest

# Run a specific test file
pytest tests/test_graph_end_to_end.py
```

## Configuration
Configuration is managed via `config.toml`. The system looks for it in `~/.deepresearch/config/config.toml` or as specified by the `--config-root` CLI flag. On the first run, it will bootstrap a default configuration in the home directory.

Key configuration areas:
- `[model]`: Ollama model names, context window sizes, and temperatures.
- `[search]`: Search backend (Tavily/DuckDuckGo) and API keys.
- `[browser]`: Docker image and timeout settings.
- `[runtime]`: Max iterations and retry logic.

## Coding Conventions
- **Type Safety:** Strict use of Pydantic for state and configuration.
- **Traceability:** Every piece of evidence MUST include a source URL and citation locator.
- **Telemetry:** All major node actions should be recorded in the state's `telemetry` list for auditability.
- **Prompts:** All LLM prompts MUST be stored in `prompts/` as Jinja2 templates.
