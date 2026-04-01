# Copilot Instructions for DeepResearch

## Commands

```bash
# Development install
pip install -e ".[dev]"

# Run the full test suite
python -m pytest

# Run a single test file
python -m pytest tests/test_graph_end_to_end.py

# Run a single test
python -m pytest tests/test_tools.py::test_duckduckgo_lite_parser_extracts_candidates

# Run the CLI locally
deepresearch "Your research query here"
```

Use `./.venv/bin/python -m pytest` if the repository virtualenv is active but `pytest` is not on `PATH`.

## Architecture

DeepResearch is a LangGraph-based research pipeline assembled in `deepresearch/main.py` and `deepresearch/graph.py`.

The CLI loads `ResearchConfig`, applies CLI overrides, builds a `ResearchRuntime`, constructs an initial `ResearchState`, then invokes the compiled graph.

`ResearchRuntime` is the dependency container for the graph. It wires together:

- `LLMWorkers` for planner / extractor / evaluator / synthesizer calls
- `LightpandaDockerManager` for page fetches through Docker
- a search backend (`DuckDuckGoSearchClient` or `TavilySearchClient`)
- `ContextManager` for token-budgeted node inputs
- `TelemetryRecorder` for auditable execution events

The state machine is:

`planner -> source_manager -> browser -> extractor -> context_manager -> evaluator -> ... -> synthesizer`

Routing is data-driven:

- `source_manager` either pops the next queued candidate or issues new searches from gaps, search intents, and active subqueries
- `browser` classifies fetched pages as useful / partial / blocked / empty / error
- `extractor` only runs for useful or partial pages
- `evaluator` decides whether research is sufficient, whether to continue sourcing, or whether to re-plan

The core design is auditability through structured state, not free-form text passing. `deepresearch/state.py` defines the durable objects that move through the graph:

- `Subquery`
- `SearchCandidate`
- `SourceVisit`
- `AtomicEvidence`
- `Gap` and `Contradiction`
- `WorkingDossier`
- `FinalReport`
- `TelemetryEvent`

Prompting is file-based and user-editable. `ResearchConfig.load()` bootstraps `~/.deepresearch/config/` from packaged defaults, and `PromptTemplateLoader` renders Jinja templates from `<config_root>/prompts`. When changing prompt variables or prompt names, update the templates and the code together.

## Key Conventions

This codebase strongly prefers typed, validated payloads over ad hoc dictionaries. Pydantic models define config, LLM payloads, evidence objects, and reports; graph state is a `TypedDict` containing those models.

Every user-facing claim should stay traceable to `AtomicEvidence`. The synthesizer builds its source list from accumulated evidence, and the working dossier is updated from accepted evidence rather than from uncited summaries.

Nodes return partial state updates, not a brand-new full state object. Keep node outputs merge-friendly and consistent with the existing graph contract.

Node execution is instrumented with telemetry through the `record_telemetry` decorator and `TelemetryRecorder`. If you add a node or a meaningful decision point, record it instead of silently changing state.

Search and evidence handling are intentionally deterministic outside the LLM boundary. Reuse helpers in `deepresearch/core/utils.py` for URL canonicalization, deduplication, chunking, scoring, coverage checks, and dossier updates before introducing new logic.

Context construction is centralized in `deepresearch/context_manager.py`. Do not hand-roll prompt context inside nodes when an existing context builder can be extended.

Config and prompts are local-first and editable by the end user. Prefer adding validated config fields in `deepresearch/config.py` and consuming them through `ResearchRuntime` instead of scattering hard-coded runtime defaults.

Tests lean on dependency injection with fake search, browser, and LLM workers (`tests/test_graph_end_to_end.py`). Preserve that seam when refactoring runtime wiring or node behavior.
