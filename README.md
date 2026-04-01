# DeepResearch 🔍

**DeepResearch** is an autonomous and auditable research agent for complex, open-ended questions. Instead of producing an answer from a single retrieval pass, it breaks the problem into subqueries, searches the web, browses candidate sources, extracts evidence, evaluates coverage, and synthesizes a report backed by traceable citations. The system is designed to run with local LLMs through Ollama.

## ✨ Key Features

- **Autonomous Research:** Decomposes complex queries into subqueries and search intents.
- **Local-First:** Optimized for local models (like Qwen 2.5 or Llama 3) via Ollama. No data leaves your machine except for web searches.
- **Web-Scale Browsing:** Uses [Lightpanda](https://lightpanda.io/), a high-performance headless browser, to navigate and extract content from the real web.
- **Traceable & Auditable:** Every claim in the final report is backed by atomic evidence, specific URLs, and direct quotations.
- **Iterative Refinement:** Evaluates its own progress, identifies knowledge gaps or contradictions, and performs follow-up searches.
- **Rich Output:** Generates structured Markdown, professional PDF reports, or sends findings directly to Discord.
- **Agentic Skill:** Compatible with agents like OpenClaw and Gemini CLI as a specialized research skill.

## 🚀 Getting Started

### 📋 Prerequisites

1. **Python 3.11+**
2. **Docker:** Required to run the Lightpanda browser instance used for web browsing and extraction.
3. **Ollama:** Installed and running with a compatible local model such as `qwen2.5:7b`.

### 📦 Installation

#### Option A: Global Install (Recommended for users)

The easiest way to install DeepResearch is using [pipx](https://github.com/pypa/pipx), which handles the virtual environment for you:

```bash
pipx install git+https://github.com/yourusername/deepresearch.git
```

#### Option B: Development Install

If you want to contribute or modify the source code:

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/deepresearch.git
cd deepresearch
```

1. **Set up a virtual environment and install in editable mode:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

1. **Pull the browser image:**

```bash
docker pull lightpanda/browser:nightly
```

## 🛠 Usage

You can run the research pipeline directly from your terminal:

```bash
# Basic research (outputs report.md)
deepresearch "What are the latest breakthroughs in solid-state battery technology in 2024?"

# Inspect planner/evaluator/extractor decisions while the run progresses
deepresearch "Compare Lightpanda vs Playwright for LLM-based web scraping" --verbosity 2

# Inspect dossier snapshots and per-page processing details
deepresearch "How mature is commercial fusion in 2026?" --verbosity 3

# Generate a PDF report
deepresearch "Compare Lightpanda vs Playwright for LLM-based web scraping" --pdf comparative_analysis.pdf

# Send the final report to Discord
deepresearch "The impact of Llama 3 on local AI" --discord
```

### ⌨️ CLI Arguments

| Argument | Description |
| :--- | :--- |
| `query` | The open-ended research question (required). |
| `--markdown PATH` | Save the final report as a Markdown file at the specified path. |
| `--pdf PATH` | Save the final report as a PDF file at the specified path. |
| `--discord` | Send the final report and executive summary to a configured Discord user via DM. |
| `--model NAME` | Override the default Ollama model (e.g., `qwen2.5:14b`). |
| `--num-ctx N` | Override the LLM context window size (default: 100,000+). |
| `--max-iterations N` | Limit the number of research cycles (default: 8). |
| `--config-root PATH` | Path to an editable configuration directory. |
| `--verbosity {0,1,2,3}` | Control debug output in `stderr` and the final `telemetry` state. `0` disables telemetry, `1` keeps the current progress logs, `2` adds LLM orchestration outputs, and `3` also adds dossier snapshots and per-page processing details. |

### Debug Verbosity Levels

Use `--verbosity` when you want to inspect how the research process evolves.

- `0`: no telemetry events are emitted to `stderr` or persisted in the final `telemetry` state.
- `1`: current progress telemetry. You see the main graph decisions such as planning, source discovery, navigation, extraction, evaluation, and synthesis progress.
- `2`: adds structured LLM orchestration outputs for planner, extractor, evaluator, and JSON repair paths so you can inspect how search, evidence, and sufficiency decisions are being made.
- `3`: adds partial dossier and coverage snapshots plus per-web processing details such as candidate ranking, page classification, excerpts, selected chunks, accepted evidence, and synthesis budget snapshots.

The debug information is available in two places:

- `stderr` during execution for live inspection.
- `final_state["telemetry"]` after graph execution when you use the graph programmatically.

## 🤖 Skill Integration (OpenClaw / Gemini CLI)

DeepResearch can be used as a "skill" by AI agents. When activated, the agent uses the `deepresearch` command to perform exhaustive background research before responding to the user.

**Example Agent Command:**

```bash
# The agent will run this in the background
deepresearch "Comprehensive analysis of solid-state battery tech and its competitors" --discord
```

## ⚙️ Configuration

On first run, DeepResearch creates an editable configuration directory at `~/.deepresearch/config/`.

### Language Support

You can configure the language for the final research report in `config.toml`. By default, it's set to English, but you can change it to your preferred language (e.g., "Spanish", "French", "German"):

```toml
[runtime]
verbosity = 0
language = "Spanish"
```

### Research depth, stagnation, and stop conditions

DeepResearch combines a hard iteration guardrail with progress-based stopping criteria.

- `max_iterations` remains the absolute safety cap.
- The agent also tracks useful progress such as newly accepted evidence, useful sources, resolved subqueries, and actionable gaps.
- Planner and evaluator prompts now look at structural coverage too: they infer the type of answer the user is asking for and watch for thin facets or overly concentrated sources.
- If the system stops making progress for too many cycles, it can stop early with `research_exhausted` instead of wasting all remaining iterations.

The main terminal stop reasons are:

- `sufficient_information`: the evaluator believes the evidence is sufficient.
- `final_context_full`: the final synthesis step is already at capacity.
- `research_exhausted`: the system is no longer making useful progress.
- `max_iterations_reached`: the hard iteration cap was reached.

These thresholds are configurable in `config.toml`:

```toml
[runtime]
max_iterations = 10
verbosity = 0
max_stagnation_cycles = 4
max_consecutive_technical_failures = 3
max_cycles_without_new_evidence = 3
max_cycles_without_useful_sources = 4
min_progress_score_to_reset_stagnation = 2

weight_new_evidence = 2
weight_useful_source = 1
weight_resolved_subquery = 3
weight_actionable_gap = 1
```

Interpretation:

- Higher `max_*` values make the agent more persistent.
- Lower values make it stop earlier when research quality is not improving.
- The `weight_*` settings control what counts as meaningful progress during a cycle.
- Better reports also depend on prompt guidance: the planner should expand under-covered facets, and the evaluator should resist stopping when the evidence is still one-sided.

### Discord Setup

To use the `--discord` flag, update `config.toml` with your bot credentials:

```toml
[discord]
token = "YOUR_BOT_TOKEN"
user_id = "YOUR_DISCORD_USER_ID"
output = "pdf" # or "markdown"
```

### Search Backend

You can switch between search providers in `config.toml`:

- **Tavily (Recommended):** High-quality search for LLMs. Requires an `api_key`.
- **DuckDuckGo:** Free, lite backend with no API key required.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
