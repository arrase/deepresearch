# Deep Research

Deep Research is an auditable research pipeline built with LangGraph, ChatOllama, and Lightpanda. It turns an open-ended question into a bounded research loop with explicit state, deterministic context selection, atomic evidence extraction, coverage evaluation, and a final cited report.

## Key Properties

- LLM backend: ChatOllama only.
- Browser backend: Lightpanda via Docker.
- State management: explicit LangGraph state, not conversation replay.
- Prompt management: editable Jinja templates loaded from the project config directory.
- Configuration: TOML loaded from config/ by default, with CLI override support for development and tests.
- Search backend default: DuckDuckGo Lite HTML parsing.
- Runtime language: English for prompts, internal chain context, telemetry, and generated markdown reports.

## Project Config Directory

The default editable configuration root is:

- config/

This directory is part of the project and contains:

- config/config.toml
- config/prompts/planner/system.j2
- config/prompts/planner/human.j2
- config/prompts/extractor/system.j2
- config/prompts/extractor/human.j2
- config/prompts/evaluator/system.j2
- config/prompts/evaluator/human.j2
- config/prompts/synthesizer/system.j2
- config/prompts/synthesizer/human.j2
- config/prompts/repair/system.j2
- config/prompts/repair/human.j2

You can override the config root at runtime with --config-root. When the override path is empty, the runtime bootstraps it from the project config directory.

## Prompt Customization

All LLM prompts are loaded from files under the selected config root. Templates use Jinja and can interpolate runtime variables such as:

- permanent
- strategic
- operational
- evidentiary
- local_source
- query
- format_instructions
- original_prompt
- raw_output
- parse_error

Prompt wording is user-editable. Schema validation, retry logic, JSON repair, and deterministic context assembly remain in Python.

## Configuration File

The main configuration file is:

- config/config.toml

It controls:

- model settings
- context window policy
- browser settings
- search settings
- runtime limits
- prompt directory and template strictness

## CLI Usage

Run a research session:

```bash
deepresearch "Is OpenAI viable in the long term?"
```

Run and specify an output file:

```bash
deepresearch "Is OpenAI viable in the long term?" -o my_report.md
```

Run with a custom config root for development:

```bash
deepresearch "Is fusion commercially viable?" --config-root ./tmp-config
```

Override selected configuration values from the CLI:

```bash
deepresearch "Is fusion commercially viable?" --config-root ./tmp-config --model qwen3.5:9b --num-ctx 32768 --max-iterations 6
```

## Development Notes

- Prompts are user-owned assets, so edit templates in config/ or in the overridden config root instead of editing inline strings in Python.
- Deterministic workers remain responsible for URL normalization, source scoring, evidence selection, deduplication, and report rendering.
- The context manager still assembles minimal context in code; only prompt wording moved to Jinja templates.

## Test Suite

Run the tests with:

```bash
pytest
```
