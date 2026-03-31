"""CLI entrypoint for the deep research system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .config import ResearchConfig
from .context_manager import ContextManager
from .graph import build_graph
from .nodes import ResearchRuntime
from .state import build_initial_state
from .subagents.llm import LLMWorkers
from .telemetry import TelemetryRecorder
from .tools import DuckDuckGoSearchClient, LightpandaDockerManager, TavilySearchClient


def build_runtime(config: ResearchConfig, verbose: bool = False) -> ResearchRuntime:
    telemetry = TelemetryRecorder(
        echo_to_console=verbose,
    )
    if config.search.backend == "tavily":
        search_client = TavilySearchClient(config.search)
    else:
        search_client = DuckDuckGoSearchClient(config.search)

    return ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=LLMWorkers(config),
        browser=LightpandaDockerManager(config.browser),
        search_client=search_client,
        telemetry=telemetry,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditable deep research with LangGraph, Ollama, and Lightpanda")
    parser.add_argument("query", help="Open-ended research question")
    parser.add_argument("-o", "--output", help="Path to write the final markdown report", default="report.md")
    parser.add_argument("--config-root", default=None, help="Path to the editable config root")
    parser.add_argument("--model", default=None, help="Ollama model name override")
    parser.add_argument("--num-ctx", type=int, default=None, help="Context window size override")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max research iterations override")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose console telemetry")

    return parser.parse_args()


def apply_cli_overrides(config: ResearchConfig, args: argparse.Namespace) -> None:
    if args.model is not None:
        config.model.model_name = args.model
    if args.num_ctx is not None:
        config.model.num_ctx = args.num_ctx
        config.context.target_tokens = args.num_ctx
    if args.max_iterations is not None:
        config.runtime.max_iterations = args.max_iterations


def cli() -> int:
    args = parse_args()
    config = ResearchConfig.load(config_root=args.config_root)
    apply_cli_overrides(config, args)

    runtime = build_runtime(config, verbose=args.verbose)
    if args.verbose:
        print(
            (
                f"Starting deep research with model={config.model.model_name}, "
                f"num_ctx={config.model.num_ctx}, max_iterations={config.runtime.max_iterations}, "
                f"config_root={config.config_root}"
            ),
            file=sys.stderr,
            flush=True,
        )

    initial_state = build_initial_state(
        args.query,
        max_iterations=config.runtime.max_iterations,
        target_tokens=config.context.target_tokens,
    )
    graph = build_graph(runtime)
    final_state = graph.invoke(initial_state)
    final_report = final_state.get("final_report")

    if final_report is None:
        print(json.dumps({"error": "Failed to generate final report"}, indent=2, ensure_ascii=True))
        return 2

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_report.markdown_report, encoding="utf-8")

    if args.verbose:
        print(
            f"Final report generated and saved to: {output_path}",
            file=sys.stderr,
            flush=True,
        )
    print(final_report.executive_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
