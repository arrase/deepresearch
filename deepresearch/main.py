"""CLI entrypoint for the deep research system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ResearchConfig
from .context_manager import ContextManager
from .core.llm import LLMWorkers
from .graph import build_graph
from .output_utils import generate_pdf
from .runtime import ResearchRuntime
from .state import build_initial_state
from .telemetry import TelemetryRecorder
from .tools import DuckDuckGoSearchClient, LightpandaDockerManager, TavilySearchClient


def build_runtime(config: ResearchConfig, verbosity: int | None = None) -> ResearchRuntime:
    active_verbosity = config.runtime.verbosity if verbosity is None else verbosity
    telemetry = TelemetryRecorder(verbosity=active_verbosity)
    if config.search.backend == "tavily":
        search_client = TavilySearchClient(config.search)
    else:
        search_client = DuckDuckGoSearchClient(config.search)

    return ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=LLMWorkers(config, telemetry=telemetry),
        browser=LightpandaDockerManager(config.browser),
        search_client=search_client,
        telemetry=telemetry,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditable deep research with LangGraph, Ollama, and Lightpanda")
    parser.add_argument("query", help="Open-ended research question")

    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--markdown", help="Path to write the final markdown report")
    output_group.add_argument("--pdf", help="Path to write the final PDF report")

    parser.add_argument("--config-root", default=None, help="Path to the editable config root")
    parser.add_argument("--model", default=None, help="Ollama model name override")
    parser.add_argument("--num-ctx", type=int, default=None, help="Context window size override")
    parser.add_argument("--max-iterations", type=int, default=None, help="Max research iterations override")
    parser.add_argument(
        "--verbosity",
        dest="verbosity",
        type=int,
        choices=range(0, 4),
        default=None,
        help="Debug verbosity level: 0 disables telemetry, 1 keeps current progress logs, 2 adds LLM orchestration outputs, 3 adds dossier snapshots and per-web processing details",
    )
    parser.add_argument("--discord", action="store_true", help="Send the final report to Discord")

    return parser.parse_args()


def apply_cli_overrides(config: ResearchConfig, args: argparse.Namespace) -> None:
    if args.model is not None:
        config.model.model_name = args.model
    if args.num_ctx is not None:
        config.model.num_ctx = args.num_ctx
    if args.max_iterations is not None:
        config.runtime.max_iterations = args.max_iterations
    if getattr(args, "verbosity", None) is not None:
        config.runtime.verbosity = args.verbosity


def cli() -> int:
    args = parse_args()
    config = ResearchConfig.load(config_root=args.config_root)
    apply_cli_overrides(config, args)

    runtime = build_runtime(config)
    if config.runtime.verbosity >= 1:
        print(
            (
                f"Starting deep research with model={config.model.model_name}, "
                f"num_ctx={config.model.num_ctx}, max_iterations={config.runtime.max_iterations}, "
                f"verbosity={config.runtime.verbosity}, config_root={config.config_root}"
            ),
            file=sys.stderr,
            flush=True,
        )

    initial_state = build_initial_state(
        args.query,
        max_iterations=config.runtime.max_iterations,
    )
    graph = build_graph(runtime)
    final_state = graph.invoke(initial_state)
    final_report = final_state.get("final_report")

    if final_report is None:
        print(json.dumps({"error": "Failed to generate final report"}, indent=2, ensure_ascii=True))
        return 2

    # File output logic:
    # 1. If --markdown is explicitly provided, use it.
    # 2. If --pdf is explicitly provided, use it.
    # 3. If NEITHER --markdown NOR --pdf is provided AND NOT --discord, default to markdown report.md.
    # 4. If --discord IS used and no other output is specified, DO NOT write to disk.
    if args.markdown:
        output_path = Path(args.markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_report.markdown_report, encoding="utf-8")
        if config.runtime.verbosity >= 1:
            print(f"Final markdown report generated and saved to: {output_path}", file=sys.stderr, flush=True)
    elif args.pdf:
        output_path = Path(args.pdf)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generate_pdf(final_report.markdown_report, output_path)
        if config.runtime.verbosity >= 1:
            print(f"Final PDF report generated and saved to: {output_path}", file=sys.stderr, flush=True)
    elif not args.discord:
        output_path = Path("report.md")
        output_path.write_text(final_report.markdown_report, encoding="utf-8")
        if config.runtime.verbosity >= 1:
            print(f"Final markdown report generated and saved to: {output_path}", file=sys.stderr, flush=True)

    if args.discord:
        import asyncio

        from .outputs.discord import send_discord_report

        print("Sending report to Discord...", file=sys.stderr, flush=True)
        success = asyncio.run(send_discord_report(config.discord, final_report))

        if success:
            print("Report sent to Discord successfully.", file=sys.stderr, flush=True)
        else:
            print("Failed to send report to Discord. Check your configuration.", file=sys.stderr, flush=True)

    print(final_report.executive_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
