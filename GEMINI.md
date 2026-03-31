# Deep Research Project Overview

`deepresearch` is an auditable deep research pipeline that leverages LangGraph, Ollama, and Lightpanda to perform comprehensive, multi-step research on open-ended queries. It is designed to be transparent, traceable, and capable of running entirely locally.

## Core Technologies
- **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) for stateful, multi-actor applications.
- **LLM:** [Ollama](https://ollama.com/) for local model execution (e.g., Qwen 2.5, Llama 3).
- **Browsing:** [Lightpanda](https://lightpanda.io/) (via Docker) for high-performance, scriptable web browsing.
- **Search:** [Tavily](https://tavily.com/) (recommended) or DuckDuckGo for web discovery.
- **Reporting:** Jinja2 for prompt templating and WeasyPrint for PDF generation.

## Architecture
The system operates as a `StateGraph` (defined in `graph.py`) with the following modular nodes:

1.  **Planner:** Analyzes the initial query and breaks it down into subqueries, search intents, and hypotheses.
2.  **Source Manager:** Prioritizes search queries and manages a queue of candidate URLs.
3.  **Browser:** Navigates to selected URLs using a Docker-managed Lightpanda instance.
4.  **Extractor:** Uses the LLM to extract atomic evidence from page content, mapped to specific subqueries.
5.  **Context Manager:** Integrates new evidence into a working dossier, performing deduplication and summarization.
6.  **Evaluator:** Assesses research coverage, identifies contradictions or gaps, and decides whether to continue or synthesize.
7.  **Synthesizer:** Generates the final executive answer and structured report with citations.

## Project Structure
- `deepresearch/`: Core source code.
    - `graph.py`: StateGraph assembly and routing logic.
    - `runtime.py`: Runtime dependencies and container (`ResearchRuntime`).
    - `nodes/`: Modular implementation of research nodes.
        - `base.py`: Shared utilities and the `@record_telemetry` decorator.
        - `planner.py`, `browser.py`, `extractor.py`, etc.: Individual node logic.
    - `core/`: Central logic and providers.
        - `llm.py`: Ollama integration, structured parsing, and payload normalization.
        - `utils.py`: Deterministic workers (deduplication, scoring, text splitting).
    - `outputs/`: Output formatting and notifications.
        - `discord.py`: Discord DM integration for reports.
    - `state.py`: Typed state and evidence models using Pydantic.
    - `config.py`: Configuration management (Pydantic settings).
    - `tools.py`: Implementation of search (Tavily/DDG) and browser (Lightpanda) clients.
- `config/`: Default configuration and prompt templates.
- `prompts/`: Jinja2 templates for each research stage (Planner, Extractor, etc.).
- `tests/`: Comprehensive test suite using `pytest`.

## Development Commands

### Environment Setup
The project requires Python 3.11+ and a running Docker daemon.
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Ensure Lightpanda image is available
docker pull lightpanda/browser:nightly
```

### Running Research
```bash
# Run a research query (defaults to report.md)
deepresearch "Future of solid-state batteries in 2024"

# Save as PDF
deepresearch "Lightpanda vs Playwright" --pdf report.pdf

# Verbose mode for real-time telemetry
deepresearch "Quantum computing breakthroughs" -v
```

### Testing
```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_graph_end_to_end.py
```

## Development Conventions
- **Modular Nodes:** Every node in the graph MUST be a class in `deepresearch/nodes/` and use the `@record_telemetry` decorator for auditing.
- **Type Safety:** Strict use of Pydantic for state (`ResearchState`) and configuration (`ResearchConfig`).
- **Traceability:** Every piece of `AtomicEvidence` MUST include a source URL, title, and citation locator.
- **Telemetry:** All major node actions and results are recorded in the state's `telemetry` list for transparency.
- **Prompts:** All LLM prompts MUST be stored in `prompts/` as Jinja2 templates and loaded via `PromptTemplateLoader`.
- **Core/Utils Separation:** Open-ended LLM logic belongs in `core/llm.py`; mechanical/deterministic tasks belong in `core/utils.py`.
