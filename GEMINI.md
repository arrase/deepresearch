# DeepResearch 🔍

DeepResearch is an autonomous, auditable research agent that performs deep-dives into complex, open-ended questions. It uses a graph-based agentic workflow to iteratively plan, search, browse, and evaluate evidence, producing comprehensive reports with full traceability.

## 🚀 Project Overview

- **Core Technology:** [LangGraph](https://github.com/langchain-ai/langgraph) for the agentic orchestration.
- **LLM Integration:** Optimized for **local models** (e.g., Qwen 2.5, Llama 3) via **Ollama**.
- **Web Browsing:** Uses [Lightpanda](https://lightpanda.io/), a high-performance headless browser (running in Docker), for web scraping.
- **Search Backends:** Supports **Tavily** (optimized for LLMs) and **DuckDuckGo** (free/lite).
- **Architecture:** A stateful `StateGraph` that manages subqueries, evidence extraction, and iterative refinement.

## 🛠 Building and Running

### Prerequisites
- **Python 3.11+**
- **Docker:** Required for the Lightpanda browser (`docker pull lightpanda/browser:nightly`).
- **Ollama:** Running locally with your target model (e.g., `ollama pull qwen2.5:7b`).

### Installation
```bash
# Clone and install in editable mode with dev dependencies
git clone <repository_url>
cd deepresearch
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Execution
```bash
# Basic CLI usage (outputs report.md)
deepresearch "Your research query here"

# Specialized outputs
deepresearch "Query" --pdf my_report.pdf
deepresearch "Query" --discord  # Requires discord configuration
```

### Testing
```bash
# Run the test suite
pytest

# Run a specific test file
pytest tests/test_graph_end_to_end.py
```

## 🏗 System Architecture

### Research State (`deepresearch/state.py`)
The system relies on a highly structured `ResearchState` (TypedDict) which tracks:
- **Active/Resolved Subqueries:** The decomposed research agenda.
- **Atomic Evidence:** Claims backed by specific URLs, quotations, and citations.
- **Visited/Discarded Sources:** Full audit trail of the browsing history.
- **Working Dossier:** Intermediate summaries and key points.
- **Telemetry:** Real-time event log of the agent's internal reasoning.

### Graph Nodes (`deepresearch/nodes/`)
1.  **Planner:** Decomposes the main query into subqueries and search intents.
2.  **Source Manager:** Selects the best URLs from search results to visit.
3.  **Browser:** Navigates to pages using Lightpanda and extracts raw content.
4.  **Extractor:** Pulls "Atomic Evidence" from page content based on active subqueries.
5.  **Context Manager:** Updates the internal dossier and manages LLM context windows.
6.  **Evaluator:** Determines if the research is sufficient or identifies "Gaps" and "Contradictions".
7.  **Synthesizer:** Generates the final Markdown report from the collected evidence.

### LLM Workers (`deepresearch/core/llm.py`)
- Uses `ChatOllama` for inference.
- Implements **Pydantic-based structured output** parsing.
- Includes a **Repair & Salvage** layer to handle imperfect JSON or truncated responses from smaller local models.

## ⚙️ Configuration

Configuration is managed via Pydantic models in `deepresearch/config.py` and stored in `~/.deepresearch/config/config.toml`.
- **Prompts:** Jinja2 templates located in `deepresearch/resources/prompts/`. Users can override these by editing the files in their local config root.
- **Models:** Default model is `qwen2.5:7b` (configurable).
- **Limits:** Configurable `max_iterations`, context window ratios, and search result counts.

## 📝 Development Conventions

- **Auditability First:** Every claim in the final output must be linked to an `AtomicEvidence` object.
- **Local-First:** Avoid dependencies on cloud-only LLMs; ensure the pipeline remains performant on consumer hardware.
- **Type Safety:** Use Pydantic for all data models and internal payloads.
- **Telemetry:** Always record significant state changes or LLM decisions via the `TelemetryRecorder`.
