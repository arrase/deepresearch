# DeepResearch

DeepResearch is a local-first research CLI for complex, open-ended questions. It plans a research path, searches the web through Tavily, extracts evidence with a local Ollama model, evaluates whether coverage is good enough, and produces a final report with traceable citations.

The project is intentionally simple: Tavily is the only search backend, and the pipeline extracts directly from Tavily raw content instead of maintaining a separate browser layer.

## Key Features

- Autonomous research planning with subqueries and search intents.
- Local-first execution through Ollama.
- Tavily-backed source discovery with raw-content extraction.
- Traceable reports backed by atomic evidence, URLs, and quotations.
- Iterative refinement that re-searches when coverage is weak.
- Markdown, PDF, and Discord output modes.

## How A Run Works

Each run follows the same loop:

1. The planner turns your question into subqueries and search intents.
2. The source manager searches Tavily and filters candidates with usable raw content.
3. The extractor turns useful passages into atomic evidence.
4. The context manager curates evidence and updates the working dossier.
5. The evaluator decides whether the research is sufficient or whether another cycle is needed.
6. The synthesizer writes the final report once the run stops.

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
- prompts/ templates for planner, extractor, evaluator, synthesizer, and repair flows

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
model_name = "qwen3.5:9b"
base_url = "OLLAMA_BASE_URL"
num_ctx = 100000
num_predict = 8192
timeout_seconds = 120
```

### Search settings

```toml
[search]
api_key = ""
results_per_query = 5
max_raw_content_chars = 24000
min_source_chars = 300
```

### Runtime settings

```toml
[runtime]
max_iterations = 8
search_batch_size = 3
min_attempts_before_exhaustion = 3
max_cycles_without_new_evidence = 4
max_cycles_without_useful_sources = 4
max_consecutive_technical_failures = 3
semantic_eval_interval = 0
allow_dynamic_replan = true
verbosity = 0
language = "English"
```

### LangSmith settings

```toml
[langsmith]
enabled = false
tracing = true
endpoint = ""
api_key = ""
project = "DeepResearch"
```

## Runtime Tuning And Stop Conditions

The hard cap for research depth is max_iterations, but the run can stop earlier when the evaluator decides that continuing is not useful.

The main stop reasons are:

- sufficient_information
- final_context_full
- research_exhausted
- max_iterations_reached

The runtime tracks signals such as newly accepted evidence, useful sources, resolved subqueries, and technical failures.

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
