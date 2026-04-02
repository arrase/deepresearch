# DeepResearch

DeepResearch is a local-first research CLI for complex, open-ended questions. It plans a research path, searches the web, browses candidate pages with Lightpanda, extracts evidence with a local Ollama model, evaluates whether coverage is good enough, and produces a final report with traceable citations.

It is built for users who want something more deliberate than a single retrieval pass, but still want to run the core reasoning stack on their own machine.

## ✨ Key Features

- **Autonomous Research:** Decomposes complex queries into subqueries and search intents.
- **Local-First:** Optimized for local models (like Qwen 2.5 or Llama 3) via Ollama. No data leaves your machine except for web searches.
- **Web-Scale Browsing:** Uses [Lightpanda](https://lightpanda.io/), a high-performance headless browser, to navigate and extract content from the real web.
- **Traceable & Auditable:** Every claim in the final report is backed by atomic evidence, specific URLs, and direct quotations.
- **Iterative Refinement:** Evaluates its own progress, identifies knowledge gaps or contradictions, and performs follow-up searches.
- **Rich Output:** Generates structured Markdown, professional PDF reports, or sends findings directly to Discord.
- **Agentic Skill:** Compatible with agents like OpenClaw and Gemini CLI as a specialized research skill.

## What It Is Good At

- Breaking a broad question into smaller research tasks.
- Collecting evidence from multiple web sources instead of answering from one snippet.
- Producing a report in Markdown or PDF, or delivering it through Discord.
- Letting you inspect the run with increasing debug detail when you need to understand what happened.
- Keeping configuration editable under your home directory instead of hiding it inside the package.

## Limitations And Caveats

- The first successful run depends on external services being ready: Docker, Lightpanda, Ollama, and a working search backend.
- The shipped config points to Tavily by default, so a missing API key will block startup until you change the search section.
- Language selection is configurable, but output quality depends on whether your chosen Ollama model is actually strong in that language.
- PDF generation relies on WeasyPrint and may depend on standard system libraries on minimal Linux installations.
- Real-world browsing can still fail on sites that block automation, require authentication, or depend on interactions outside the current browser flow.
- Search quality and source availability vary over time; the tool can only reason over what it can successfully discover and fetch during the run.

## Who It Is For

DeepResearch is aimed at users who already work comfortably from the terminal and want a repeatable research workflow for topics such as market analysis, technical comparisons, capability tracking, or background investigation before writing.

It is less suitable if you want a point-and-click app, guaranteed access to every website, or a hosted service with no local dependencies.

## How A Run Works

Each run follows the same high-level loop:

1. The planner turns your question into subqueries and search intents.
2. The source manager searches for candidate URLs.
3. Lightpanda opens those pages and extracts page content.
4. The extractor turns useful passages into atomic evidence.
5. The evaluator decides whether the research is sufficient or whether another cycle is needed.
6. The synthesizer writes the final report once the run stops.

The stop reason is part of the final state and typically falls into one of these buckets: enough information was gathered, the synthesis context filled up, the process stopped making useful progress, or the maximum iteration limit was reached.

## Requirements

You need all of the following before expecting a successful run:

- Python 3.11 or newer
- Docker, for the Lightpanda browser container
- Ollama running locally
- At least one Ollama model pulled locally
- A search backend configured in `~/.deepresearch/config/config.toml`

The shipped configuration uses Tavily by default. If you do not want to provide a Tavily API key, switch the backend to `duckduckgo_lite` before your first real run.

## Installation

### Option A: Install with pipx

This is the cleanest option if you want the `deepresearch` command globally available without managing a virtual environment yourself.

```bash
pipx install git+https://github.com/arrase/deepresearch.git
```

### Option B: Install from source

```bash
git clone https://github.com/arrase/deepresearch.git
cd deepresearch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Prepare The Runtime

### 1. Pull the browser image

```bash
docker pull lightpanda/browser:nightly
```

### 2. Install and start Ollama

If Ollama is not already installed, get it from https://ollama.com.

Then pull a model that matches the default configuration, or plan to override the model name later.

```bash
ollama pull qwen3.5:9b
ollama serve
```

If Ollama is already running as a background service on your machine, you do not need to start it again manually.

### 3. Bootstrap and edit the config

DeepResearch creates an editable config tree under `~/.deepresearch/config/` the first time the CLI loads its configuration. That tree includes:

- `config.toml`
- `prompts/` templates for planner, extractor, evaluator, synthesizer, and repair flows

If that directory does not exist yet, invoke the CLI once to materialize it, then edit the generated `config.toml` before expecting a successful research run.

Because the shipped `config.toml` points to Tavily by default, you should open `~/.deepresearch/config/config.toml` before your first successful research run and choose one of these paths:

- Add your Tavily API key and keep `backend = "tavily"`
- Change `backend = "duckduckgo_lite"` if you want a no-key search backend

Minimal search configuration example:

```toml
[search]
backend = "duckduckgo_lite"
api_key = ""
```

Or, if you want Tavily:

```toml
[search]
backend = "tavily"
api_key = "YOUR_TAVILY_API_KEY"
```

## First Successful Run

Once Docker, Ollama, and search configuration are ready, run a simple query:

```bash
deepresearch "Compare Lightpanda and Playwright for LLM-driven web extraction"
```

If you do not pass an explicit output flag and you are not using Discord delivery, DeepResearch writes the full report to `report.md` in the current directory and prints the executive answer to standard output.

## Command-Line Usage

The CLI shape is:

```bash
deepresearch "your question" [options]
```

### Arguments

| Argument | What it does |
| --- | --- |
| `query` | Required open-ended research question |
| `--markdown PATH` | Write the final report as Markdown |
| `--pdf PATH` | Write the final report as PDF |
| `--discord` | Send the final report to the configured Discord user |
| `--model NAME` | Override the Ollama model name for this run |
| `--num-ctx N` | Override the model context window |
| `--max-iterations N` | Override the maximum number of research cycles |
| `--config-root PATH` | Use a different editable config directory |
| `--verbosity {0,1,2,3}` | Control telemetry and debug detail |

`--markdown` and `--pdf` are mutually exclusive. `--discord` can be used on its own or combined with either of the file-output options.

### Common examples

Write Markdown to a custom path:

```bash
deepresearch "Assess the current commercial readiness of fusion startups" --markdown outputs/fusion.md
```

Generate a PDF instead of Markdown:

```bash
deepresearch "Map the tradeoffs between local browser automation stacks" --pdf outputs/browser-stack.pdf
```

Override the model and context window for one run:

```bash
deepresearch "Track the strongest open-source coding models this quarter" --model llama3.1:8b --num-ctx 65536
```

Run with live debug output:

```bash
deepresearch "Evaluate the current state of multimodal local models" --verbosity 2
```

Use a project-local config directory instead of the default home-directory config:

```bash
deepresearch "Research question" --config-root .deepresearch-config
```

## Output Behavior

DeepResearch applies output rules in a fixed order:

- If you pass `--markdown`, it writes Markdown to that path.
- If you pass `--pdf`, it writes a PDF to that path.
- If you pass neither `--markdown` nor `--pdf`, and you are not using Discord-only delivery, it writes `report.md` in the current working directory.
- If you pass only `--discord`, it sends the report to Discord and does not write a file to disk.
- If you combine `--discord` with `--markdown` or `--pdf`, it both writes the file and sends the report.

The CLI always prints the executive answer to standard output when a final report is produced.

## Verbosity Levels

Use `--verbosity` when you want insight into how the run progressed.

- `0`: no telemetry output
- `1`: graph-level progress through planning, discovery, browsing, extraction, evaluation, and synthesis
- `2`: adds LLM orchestration details and JSON-repair paths
- `3`: adds dossier snapshots and page-level processing details

Higher verbosity is useful for debugging weak results, stalled runs, or unexpected source choices.

## Configuration

The main user-editable file is `~/.deepresearch/config/config.toml`. The runtime validates it strictly, so unknown sections and unsupported fields are rejected instead of silently ignored.

These are the settings most users will care about first.

### Model settings

```toml
[model]
model_name = "qwen3.5:9b"
base_url = "http://127.0.0.1:11434"
num_ctx = 150000
num_predict = 8192
timeout_seconds = 120
```

Use this section to point DeepResearch at your Ollama server and choose the model budget you want to run.

### Browser settings

```toml
[browser]
image = "lightpanda/browser:nightly"
wait_ms = 7000
wait_until = "networkidle"
obey_robots = true
max_content_chars = 24000
```

These values control how aggressively Lightpanda waits for content and how much text is kept from each page.

### Search settings

```toml
[search]
backend = "tavily"
api_key = ""
results_per_query = 8
max_queries_per_cycle = 5
max_queue_size = 30
```

This section determines where candidate URLs come from and how wide each research cycle expands.

### Runtime settings

```toml
[runtime]
max_iterations = 10
verbosity = 0
language = "English"
eval_batch_size = 3
max_stagnation_cycles = 4
max_consecutive_technical_failures = 3
max_cycles_without_new_evidence = 3
max_cycles_without_useful_sources = 4
```

Use this section when you want to make runs shorter, more persistent, or emit reports in a different language.

## Search Backends And Model Choices

DeepResearch currently supports two search paths:

- `tavily`: stronger search quality, but requires an API key
- `duckduckgo_lite`: no API key, easier to start with, but more dependent on HTML scraping conditions

For models, the default configuration assumes an Ollama-served local model such as `qwen3.5:9b`. You can point the tool at another compatible Ollama model through `config.toml` or `--model`.

In practice, output quality depends heavily on the model you choose. Smaller models may complete the pipeline but still produce weaker planning, extraction, or synthesis than larger ones.

## Runtime Tuning And Stop Conditions

The hard cap for research depth is `max_iterations`, but the run can stop earlier when the evaluator decides that continuing is not useful.

The main stop reasons are:

- `sufficient_information`: the evaluator judged the gathered evidence to be enough
- `final_context_full`: the synthesis stage ran out of room for more context
- `research_exhausted`: the run stopped making useful progress
- `max_iterations_reached`: the configured cap was hit

The runtime tracks several signals to determine whether progress is still happening, including newly accepted evidence, useful sources, resolved subqueries, technical failures, and cycles with no meaningful improvement.

If you want deeper runs, raise `max_iterations` and the stagnation thresholds. If you want faster cutoffs, lower them.

## Discord Delivery

Discord delivery is optional and configured in `config.toml`.

```toml
[discord]
token = "YOUR_BOT_TOKEN"
user_id = "YOUR_DISCORD_USER_ID"
output = "pdf"
```

When a report is short enough, DeepResearch sends it as a direct message body. Longer reports are sent as a file attachment, either as PDF or Markdown depending on the configured `output` value.

Example:

```bash
deepresearch "Summarize the strongest evidence for local browser automation stacks" --discord
```

## Brief Agent Integration

If you use other agent frameworks or custom shells, DeepResearch fits best as a background research step that is invoked through the CLI and then consumed as a file or Discord artifact.

Typical pattern:

```bash
deepresearch "Comprehensive analysis of local coding agents and tool use patterns" --markdown artifacts/research.md
```

That keeps integration simple because the external agent only needs to launch the command and read the resulting artifact.

## License

DeepResearch is distributed under the MIT License. See `LICENSE` for details.
# DeepResearch 🔍

**DeepResearch** is an autonomous and auditable research agent for complex, open-ended questions. Instead of producing an answer from a single retrieval pass, it breaks the problem into subqueries, searches the web, browses candidate sources, extracts evidence, evaluates coverage, and synthesizes a report backed by traceable citations. The system is designed to run with local LLMs through Ollama.

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
