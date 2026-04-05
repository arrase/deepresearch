# Project Guidelines

## Architecture

- DeepResearch is a LangGraph pipeline. The runtime is assembled in `deepresearch/main.py` via `build_runtime`, and the graph is defined in `deepresearch/graph.py`.
- Keep node responsibilities separated: planner, source discovery, extractor, context manager, evaluator, and synthesizer live under `deepresearch/nodes/` and should continue to communicate through `ResearchState` updates rather than direct cross-node calls.
- Nodes should consume dependencies from `ResearchRuntime`; do not instantiate config, search, or LLM clients inside node logic.
- Prompt changes belong in `deepresearch/resources/prompts/`; update Python logic only when the contract or rendering flow changes.
- Configuration is defined by strict Pydantic models in `deepresearch/config.py`. When adding config, update both the schema and `deepresearch/resources/config.toml`.

## Build and Test

- Use Python 3.11+.
- Install dev dependencies with `pip install -e ".[dev]"`.
- Run the main validation commands before finishing substantial changes: `pytest`, `ruff check .`, and `mypy .`.
- Prefer focused test runs while iterating, then run the relevant broader suite before concluding.
- Integration-style behavior may depend on Docker, Ollama, Tavily, or Discord configuration. Do not assume those services are available in local test runs unless the task explicitly requires them.

## Code Style and Conventions

- Follow the existing Python style: `from __future__ import annotations`, typed models/state objects, and small focused functions.
- Preserve the package split already established in `deepresearch/tools/`, `deepresearch/core/`, and `deepresearch/nodes/`; avoid collapsing modules back into large utility files.
- Nodes should return partial `ResearchState` updates that remain merge-friendly rather than rebuilding the full state object.
- State transitions are controlled in `deepresearch/graph.py`. If behavior changes across stages, update routing logic and tests together.
- Context construction is centralized in `deepresearch/context_manager.py`; extend that flow instead of assembling ad hoc prompt inputs inside nodes.
- Test helpers and fake implementations belong in `tests/conftest.py` or nearby test modules. Reuse the existing deterministic fixtures pattern instead of introducing ad hoc mocks.
- When changing search, browser, or evidence logic, prefer focused tests such as `tests/test_tools.py`, `tests/test_context_manager.py`, and `tests/test_graph_end_to_end.py` instead of relying only on broad end-to-end coverage.
- Keep telemetry and error handling explicit. Follow the existing node patterns instead of using broad exception handling.
- Locale-aware reporting and prompt loading are runtime concerns; keep changes consistent with the configured `runtime.language` flow.

## Working Notes for Agents

- Read `README.md` for installation, CLI flags, output modes, and environment prerequisites instead of duplicating that content here.
- Tavily is the only supported search backend; keep the integration simple and avoid reintroducing compatibility layers for removed backends.
- The editable user config is bootstrapped under `~/.deepresearch/config/`. Preserve that workflow when changing configuration loading.
- This repo uses strict config validation and typed payload contracts. Favor root-cause fixes over permissive fallbacks that hide malformed inputs.
