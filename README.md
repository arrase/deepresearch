# DeepResearch

DeepResearch is a local-first research CLI for complex, open-ended questions. It plans a research path, searches the web through Tavily, extracts evidence with a local Ollama model, evaluates whether coverage is good enough, and produces a final report with traceable citations.

## Key Features

- Hierarchical Map-Reduce architecture that decomposes questions into independent chapters.
- Local-first execution through Ollama with per-stage temperature control.
- Tavily-backed source discovery with raw-content extraction.
- Traceable reports backed by atomic evidence, URLs, and quotations.
- Deterministic evaluator that tracks stagnation signals without LLM calls.
- Devil's advocate auditor that can reject and re-plan chapters before synthesis.
- Markdown, PDF, and Discord output modes.

## How A Run Works

Each run follows a hierarchical Map-Reduce pipeline with nine graph nodes:

1. The **meta-planner** decomposes the research question into independent chapters, each covering a distinct dimension of the topic.
2. For each chapter, the **micro-planner** creates focused sub-topics with concrete search terms and intents.
3. The **source manager** executes Tavily searches and filters candidates with enough raw content.
4. The **extractor** turns useful passages into atomic evidence objects attached to sources.
5. The **context manager** deduplicates evidence, merges near-duplicates, and updates the per-topic working dossier.
6. The **evaluator** runs a deterministic coverage assessment (no LLM) and decides whether to continue searching, move to audit, or stop.
7. Once a chapter's topics are covered, the **auditor** reviews the evidence as a devil's advocate. It can reject and send the chapter back to the micro-planner for additional research.
8. The **sub-synthesizer** produces a chapter draft from the curated evidence for each completed chapter.
9. After all chapters are done, the **global synthesizer** assembles every chapter draft into the final report with an executive answer, key findings, and traceable citations.

The Map phase researches each chapter independently through its own search-extract-evaluate loop. The Reduce phase merges all chapter drafts into a single coherent report.

## Requirements

- Python 3.11 or newer.
- Ollama running locally.
- At least one Ollama model pulled locally.
- A Tavily API key configured in ~/.deepresearch/config/config.toml.

PDF generation still relies on WeasyPrint and may require standard system libraries on minimal Linux installations.

## Installation

### Option A: pipx

```bash
pipx install git+https://github.com/arrase/deepresearch.git
```

### Option B: source install

```bash
git clone https://github.com/arrase/deepresearch.git
cd deepresearch
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Prepare The Runtime

### 1. Install and start Ollama

If Ollama is not already installed, get it from <https://ollama.com>.

```bash
ollama pull qwen3.5:9b
ollama serve
```

### 2. Bootstrap and edit the config

DeepResearch creates an editable config tree under ~/.deepresearch/config/ the first time the CLI loads its configuration. That tree includes:

- config.toml
- prompts/ templates for meta-planner, micro-planner, extractor, auditor, sub-synthesizer, global synthesizer, and repair flows

Before your first real run, set a Tavily API key in ~/.deepresearch/config/config.toml.

Minimal search configuration:

```toml
[search]
api_key = "YOUR_TAVILY_API_KEY"
results_per_query = 5
max_raw_content_chars = 24000
min_source_chars = 300
```

## First Successful Run

Once Ollama and Tavily are ready, run a simple query:

```bash
deepresearch "Compare web research APIs for deep analysis workflows"
```

If you do not pass an explicit output flag and you are not using Discord delivery, DeepResearch writes the full report to report.md in the current directory and prints the executive answer to standard output.

## Command-Line Usage

deepresearch "your question" [options]

### Arguments

| Argument | What it does |
| --- | --- |
| query | Required open-ended research question |
| --markdown PATH | Write the final report as Markdown |
| --pdf PATH | Write the final report as PDF |
| --discord | Send the final report to the configured Discord user |
| --model NAME | Override the Ollama model name for this run |
| --num-ctx N | Override the model context window |
| --max-iterations N | Override the maximum number of research cycles |
| --config-root PATH | Use a different editable config directory |
| --verbosity {0,1,2,3} | Control local progress logging detail |

### Common examples

```bash
deepresearch "Assess the current commercial readiness of fusion startups" --markdown outputs/fusion.md

deepresearch "Compare web research APIs for deep analysis workflows" --pdf outputs/research-apis.pdf

deepresearch "Track the strongest open-source coding models this quarter" --model llama3.1:8b --num-ctx 65536

deepresearch "Evaluate the current state of multimodal local models" --verbosity 2

deepresearch "Research question" --config-root .deepresearch-config
```

## Output Behavior

- If you pass --markdown, it writes Markdown to that path.
- If you pass --pdf, it writes a PDF to that path.
- If you pass neither --markdown nor --pdf, and you are not using Discord-only delivery, it writes report.md in the current working directory.
- If you pass only --discord, it sends the report to Discord and does not write a file to disk.
- If you combine --discord with --markdown or --pdf, it both writes the file and sends the report.

The CLI always prints the executive answer to standard output when a final report is produced.

## Configuration

The main user-editable file is ~/.deepresearch/config/config.toml. The runtime validates it strictly, so unknown sections and unsupported fields are rejected instead of silently ignored.

### Model settings

```toml
[model]
model_name = "qwen3.5:9b" # Ollama model name used for all research stages.
base_url = "http://127.0.0.1:11434" # Base URL of the local or remote Ollama server.
temperature_meta_planner = 0.3 # Sampling temperature for chapter decomposition.
temperature_micro_planner = 0.2 # Sampling temperature for sub-topic planning.
temperature_extractor = 0.0 # Sampling temperature for evidence extraction (deterministic).
temperature_auditor = 0.3 # Sampling temperature for the devil's advocate auditor.
temperature_sub_synthesizer = 0.1 # Sampling temperature for per-chapter synthesis.
temperature_global_synthesizer = 0.1 # Sampling temperature for final report assembly.
num_ctx = 100000 # Maximum context window passed to Ollama.
num_predict = 8192 # Maximum tokens generated per LLM call.
timeout_seconds = 120 # Per-request timeout for Ollama calls.
```

### Search settings

```toml
[search]
api_key = "" # Tavily API key used for web search requests.
results_per_query = 5 # Maximum Tavily results requested for each search query.
max_raw_content_chars = 24000 # Maximum raw page characters kept from each search result.
min_source_chars = 300 # Minimum source content length required before extraction.
```

### Runtime settings

```toml
[runtime]
max_iterations = 8 # Hard cap on total search cycles across all chapters.
search_batch_size = 3 # How many candidate search queries to execute per cycle.
min_attempts_before_exhaustion = 3 # Minimum attempts before a topic can be marked as exhausted.
max_cycles_without_new_evidence = 4 # Stop after this many cycles without newly accepted evidence.
max_cycles_without_useful_sources = 4 # Stop after this many cycles without useful sources.
max_consecutive_technical_failures = 3 # Abort after too many consecutive technical failures.
max_chapters = 5 # Maximum chapters the meta-planner can create (1-10).
max_topic_depth = 2 # Maximum nesting depth for sub-topics below a chapter (1-4).
max_audit_rejections = 2 # How many times the auditor can reject a chapter before auto-approval (0-5).
verbosity = 0 # CLI log verbosity from quiet to detailed diagnostics.
llm_retry_attempts = 2 # How many times to retry recoverable LLM parsing failures.
language = "English" # Language used for the final report.
```

### LangSmith settings

```toml
[langsmith]
enabled = false # Enable LangSmith integration for this run.
tracing = true # Emit tracing spans when LangSmith integration is enabled.
endpoint = "" # Custom LangSmith API endpoint, if you are not using the default service.
api_key = "" # LangSmith API key required when tracing is enabled.
project = "DeepResearch" # LangSmith project name used for uploaded traces.
```

### Troubleshooting invalid config

- The CLI prints the exact file path and setting when validation fails.
- Unknown sections or keys are rejected on purpose; remove or rename the unsupported setting.
- If your config came from an older release and has drifted too far, move ~/.deepresearch/config/config.toml to a backup location and run the command again to regenerate a fresh commented config.

## Runtime Tuning And Stop Conditions

The hard cap for research depth is max_iterations, but the run can stop earlier when the deterministic evaluator decides that continuing is not useful.

The evaluator tracks stagnation signals without making LLM calls: newly accepted evidence counts, useful source discovery, per-topic coverage, and consecutive technical failures.

The main stop reasons are:

- **CONTEXT_SATURATION** when the synthesis token budget has been effectively consumed.
- **PLAN_COMPLETED** when every chapter has been audited and synthesized.
- **MAX_ITERATIONS_REACHED** when the run reaches the configured hard cap.
- **STUCK_NO_SOURCES** when repeated cycles fail to produce new evidence or useful sources.

## Discord Delivery

Discord delivery is optional and configured in config.toml.

```toml
[discord]
token = "YOUR_BOT_TOKEN"
user_id = "YOUR_DISCORD_USER_ID"
output = "pdf"
```

Example:

```bash
deepresearch "Summarize the strongest evidence for research APIs" --discord
```

## License

DeepResearch is distributed under the MIT License. See LICENSE for details.
