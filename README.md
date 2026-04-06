# DeepResearch

DeepResearch is a local-first research CLI for complex, open-ended questions. It plans a research path, searches the web through Tavily, extracts evidence with a local Ollama model, evaluates whether coverage is good enough, synthesizes topic briefs, and produces a final report with traceable citations.

## Key Features

- Autonomous research planning with subqueries and search intents.
- Local-first execution through Ollama.
- Tavily-backed source discovery with raw-content extraction.
- Multi-pass extraction from the same source when the content is dense enough to justify deeper evidence capture.
- Topic-level synthesis before the final report, so long answers are built from richer intermediate briefs instead of a flat evidence list.
- Traceable reports backed by atomic evidence, URLs, and quotations.
- Iterative refinement that re-searches when coverage is weak.
- Configurable report language. The final report parser is language-agnostic as long as the generated report keeps the expected section structure.
- Markdown, PDF, and Discord output modes.

## How A Run Works

Each run follows the same loop:

1. The planner turns your question into subqueries and search intents.
2. The source manager searches Tavily and filters candidates with usable raw content.
3. The extractor runs one or more extraction passes over the most relevant chunks of each source and turns useful passages into atomic evidence.
4. The context manager curates evidence and updates the working dossier.
5. The evaluator decides whether the research is sufficient or whether another cycle is needed.
6. The topic synthesizer writes a brief for each completed topic once the run stops.
7. The synthesizer writes the final report from topic briefs, dossier state, and selected evidence.

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
- prompts/ templates for planner, extractor, evaluator, topic_synthesizer, synthesizer, and repair flows

Before your first real run, set a Tavily API key in ~/.deepresearch/config/config.toml.

Minimal search configuration:

```toml
[search]
api_key = "YOUR_TAVILY_API_KEY"
results_per_query = 3
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
temperature_planner = 0.2 # Sampling temperature for planning.
temperature_extractor = 0.0 # Sampling temperature for evidence extraction.
temperature_evaluator = 0.0 # Sampling temperature for coverage evaluation.
temperature_synthesizer = 0.1 # Sampling temperature for final report synthesis.
temperature_topic_synthesizer = 0.1 # Sampling temperature for topic-brief synthesis.
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

### Reporter settings

```toml
[reporter]
output_reserve_ratio = 0.20 # Fraction of the context window reserved for the final answer.
prompt_margin_tokens = 512 # Extra prompt headroom kept free before synthesis.
topic_brief_budget_ratio = 0.45 # Share of the synthesis prompt budget reserved for topic briefs.
final_report_target_words = 1800 # Approximate target length for rich final reports.
```

### Runtime settings

```toml
[runtime]
max_iterations = 8 # Hard cap on planner and search cycles for a run.
search_batch_size = 3 # How many candidate search queries to execute per cycle.
min_attempts_before_exhaustion = 3 # Minimum attempts before a topic can be marked as exhausted.
max_cycles_without_new_evidence = 4 # Stop after this many cycles without newly accepted evidence.
max_cycles_without_useful_sources = 4 # Stop after this many cycles without useful sources.
max_consecutive_technical_failures = 3 # Abort after too many consecutive technical failures.
semantic_eval_interval = 0 # Run evaluator every N cycles even without strong evidence updates; 0 disables it.
allow_dynamic_replan = true # Allow the planner to revise the topic plan during the run.
verbosity = 0 # CLI log verbosity from quiet to detailed diagnostics.
llm_retry_attempts = 2 # How many times to retry recoverable LLM parsing failures.
min_topic_evidence_target = 2 # Minimum evidence target applied after planner normalization.
max_topic_evidence_target = 4 # Upper bound for dynamically normalized topic evidence targets.
extraction_max_chars_per_pass = 4000 # Maximum source characters sent to one extraction pass.
max_extraction_passes_per_source = 3 # How many extraction passes can be run for one source.
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

The hard cap for research depth is max_iterations, but the run can stop earlier when the evaluator decides that continuing is not useful.

The main stop reasons are:

- context_saturation
- plan_completed
- max_iterations_reached
- stuck_no_sources

The runtime tracks signals such as newly accepted evidence, useful sources, source diversity, resolved subqueries, and technical failures.

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
