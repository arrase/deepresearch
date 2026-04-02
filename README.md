# DeepResearch 🔍

**DeepResearch**  is a local-first research CLI for complex, open-ended questions. It plans a research path, searches the web, browses candidate pages with Lightpanda, extracts evidence with a local Ollama model, evaluates whether coverage is good enough, and produces a final report with traceable citations.

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