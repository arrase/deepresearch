# Deep Research

Deep Research is an auditable, stateful research pipeline designed to transform open-ended questions into high-quality research reports with explicit citations. It uses a graph-based architecture to manage a research loop involving planning, source discovery, web browsing, evidence extraction, and synthesis.

The system is optimized for **local LLMs** (via Ollama) and features a robust "mechanical" layer for data cleanup, scoring, and deduplication, ensuring that the final output is reliable and verifiable.

## Key Features

- **Stateful Research Loop**: Built with [LangGraph](https://github.com/langchain-ai/langgraph), the system manages the research lifecycle through a series of specialized nodes (Planner, Browser, Extractor, Evaluator, Synthesizer).
- **Local LLM Support**: Powered by [Ollama](https://ollama.com/), allowing for private and cost-effective research sessions using models like Qwen 2.5 or Llama 3.
- **Auditable Reports**: Every claim in the final report is linked to an `evidence_id`, which is traced back to a specific source URL and quotation.
- **Headless Browsing**: Uses [Lightpanda](https://lightpanda.io/) (via Docker) as a high-performance headless browser for reliable web scraping and evidence extraction.
- **Robust Data Handling**: Includes a "salvage" layer for fixing imperfect JSON outputs from smaller local models and deterministic workers for URL canonicalization, deduplication, and heuristic scoring.
- **Customizable Prompts**: Prompts are stored as [Jinja2](https://jinja.palletsprojects.com/) templates in `config/prompts/`, allowing for easy fine-tuning of the research behavior without modifying the core logic.

## Architecture Overview

The research process follows a directed graph:

1.  **Planner**: Analyzes the initial query and generates a research agenda with subqueries and search intents.
2.  **Source Manager**: Discovers and prioritizes candidate URLs based on the current research gaps.
3.  **Browser**: Fetches and cleans content from candidate pages using a Docker-managed Lightpanda instance.
4.  **Extractor**: Pulls atomic evidence (claims, quotations, citations) from the fetched content.
5.  **Context Manager**: Assembles the "Working Dossier" and manages the token budget for the LLM.
6.  **Evaluator**: Assesses research coverage, identifies contradictions, and decides if more iterations are needed.
7.  **Synthesizer**: Generates a comprehensive Markdown report with an executive answer and referenced sources.

## Prerequisites

- **Python**: 3.11 or higher.
- **Ollama**: Installed and running (for the LLM backend).
- **Docker**: Installed and running (for the Lightpanda browser).

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-repo/deepresearch.git
    cd deepresearch
    ```

2.  Install the package in editable mode:
    ```bash
    pip install -e .
    ```

3.  Ensure you have the required Ollama model (default is `qwen3.5:9b` or similar):
    ```bash
    ollama pull qwen3.5:9b
    ```

## Usage

Run a research session by providing a query:

```bash
deepresearch "What are the latest developments in fusion energy for 2025?"
```

### CLI Options

- `-o, --output <path>`: Path to save the final Markdown report (default: `report.md`).
- `--model <name>`: Override the default Ollama model name.
- `--config-root <path>`: Use a custom configuration directory.
- `--max-iterations <int>`: Set the maximum number of research cycles.
- `--discord`: Send the final report to a Discord user via DM (requires configuration).
- `-v, --verbose`: Enable detailed console telemetry to follow the research process in real-time.

## Configuration

The project uses a TOML-based configuration system. On the first run, it will bootstrap a default configuration in `~/.deepresearch/config`.

- **`config.toml`**: Manage model parameters (temperature, context window), browser settings, and research limits.
- **`discord` section**: Add your `token` and `user_id` to enable Discord notifications.
- **`prompts/`**: Edit the Jinja2 templates to change how the LLM plans, extracts evidence, or synthesizes the final report.

## Development

### Project Structure

- `deepresearch/`: Core Python package.
  - `graph.py`: LangGraph workflow definition.
  - `nodes.py`: Node implementations (logic for each research step).
  - `state.py`: Typed data models and Pydantic schemas.
  - `subagents/`: Specialized workers (LLM reasoning and deterministic data processing).
  - `context_manager.py`: Context assembly and token budgeting.
- `config/`: Default configuration and prompt templates.
- `tests/`: Comprehensive test suite using Pytest.
