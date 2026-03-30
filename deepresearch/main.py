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
from .tools import DuckDuckGoSearchClient, LightpandaDockerManager, self_check_services


def build_runtime(config: ResearchConfig) -> ResearchRuntime:
    config.ensure_directories()
    telemetry = TelemetryRecorder(
        artifacts_dir=config.runtime.artifacts_dir,
        logs_dir=config.runtime.logs_dir,
        echo_to_console=True,
    )
    return ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=LLMWorkers(config),
        browser=LightpandaDockerManager(config.browser),
        search_client=DuckDuckGoSearchClient(config.search),
        telemetry=telemetry,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditable deep research with LangGraph, Ollama, and Lightpanda")
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run", help="Run a research session")
    run_parser.add_argument("query", help="Open-ended research question")
    run_parser.add_argument("--config-root", default=None, help="Path to the editable config root")
    run_parser.add_argument("--model", default=None)
    run_parser.add_argument("--num-ctx", type=int, default=None)
    run_parser.add_argument("--max-iterations", type=int, default=None)
    run_parser.add_argument("--artifacts-dir", default=None)
    run_parser.add_argument("--logs-dir", default=None)
    run_parser.add_argument("--skip-self-check", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")

    check_parser = subparsers.add_parser("self-check", help="Validate Ollama, Docker, and Lightpanda")
    check_parser.add_argument("--config-root", default=None, help="Path to the editable config root")
    check_parser.add_argument("--model", default=None)
    check_parser.add_argument("--num-ctx", type=int, default=None)

    return parser.parse_args()


def apply_cli_overrides(config: ResearchConfig, args: argparse.Namespace) -> None:
    applied_override = False
    if getattr(args, "model", None) is not None:
        config.model.model_name = args.model
        applied_override = True
    if getattr(args, "num_ctx", None) is not None:
        config.model.num_ctx = args.num_ctx
        config.context.target_tokens = args.num_ctx
        applied_override = True
    if getattr(args, "max_iterations", None) is not None:
        config.runtime.max_iterations = args.max_iterations
        applied_override = True
    if getattr(args, "artifacts_dir", None) is not None:
        config.runtime.artifacts_dir = Path(args.artifacts_dir)
        applied_override = True
    if getattr(args, "logs_dir", None) is not None:
        config.runtime.logs_dir = Path(args.logs_dir)
        applied_override = True
    if getattr(args, "config_root", None) is not None or applied_override:
        config.context.configured_by = "cli_override"


def cli() -> int:
    args = parse_args()
    command = args.command or "run"
    config = ResearchConfig.load(config_root=getattr(args, "config_root", None))
    apply_cli_overrides(config, args)

    runtime = build_runtime(config)
    if getattr(args, "quiet", False):
        runtime.telemetry._echo_to_console = False
    else:
        print(
            (
                f"Starting deep research with model={config.model.model_name}, "
                f"num_ctx={config.model.num_ctx}, max_iterations={config.runtime.max_iterations}, "
                f"config_root={config.config_root}"
            ),
            file=sys.stderr,
            flush=True,
        )
    report = self_check_services(browser=runtime.browser, model=config.model)
    if command == "self-check":
        print(json.dumps(report.__dict__, indent=2, ensure_ascii=True))
        return 0 if report.docker_ok and report.ollama_ok and report.model_available else 1

    if not getattr(args, "skip_self_check", False):
        print("Operational self-check completed, starting graph...", file=sys.stderr, flush=True)
        if not report.docker_ok or not report.ollama_ok or not report.model_available:
            print(json.dumps(report.__dict__, indent=2, ensure_ascii=True))
            return 1

    initial_state = build_initial_state(
        args.query,
        max_iterations=config.runtime.max_iterations,
        target_tokens=config.context.target_tokens,
        configured_by=config.context.configured_by,
        selection_policy=config.context.selection_policy,
    )
    graph = build_graph(runtime)
    final_state = graph.invoke(initial_state)
    final_report = final_state.get("final_report")
    if final_report is None:
        print(json.dumps({"error": "Failed to generate final report"}, indent=2, ensure_ascii=True))
        return 2
    markdown_path = runtime.telemetry.write_markdown_report(final_report, label="final_report")
    final_report.markdown_artifact_path = str(markdown_path)
    checkpoint_path = runtime.telemetry.checkpoint(final_state, label="final")
    print(
        f"Final report generated. Markdown: {markdown_path} | Checkpoint: {checkpoint_path}",
        file=sys.stderr,
        flush=True,
    )
    print(final_report.executive_answer)
    print(f"\nMarkdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
