"""CLI del sistema de deep research."""

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
        llm_workers=LLMWorkers(config.model, config.runtime),
        browser=LightpandaDockerManager(config.browser),
        search_client=DuckDuckGoSearchClient(config.search),
        telemetry=telemetry,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep research auditable con LangGraph, Ollama y Lightpanda")
    subparsers = parser.add_subparsers(dest="command", required=False)

    run_parser = subparsers.add_parser("run", help="Ejecuta una investigacion")
    run_parser.add_argument("query", help="Pregunta de investigacion abierta")
    run_parser.add_argument("--model", default="qwen3.5:9b")
    run_parser.add_argument("--num-ctx", type=int, default=100000)
    run_parser.add_argument("--max-iterations", type=int, default=8)
    run_parser.add_argument("--artifacts-dir", default="artifacts")
    run_parser.add_argument("--logs-dir", default="logs")
    run_parser.add_argument("--skip-self-check", action="store_true")
    run_parser.add_argument("--quiet", action="store_true")

    check_parser = subparsers.add_parser("self-check", help="Valida Ollama, Docker y Lightpanda")
    check_parser.add_argument("--model", default="qwen3.5:9b")
    check_parser.add_argument("--num-ctx", type=int, default=100000)

    return parser.parse_args()


def cli() -> int:
    args = parse_args()
    command = args.command or "run"
    config = ResearchConfig()
    config.model.model_name = getattr(args, "model", config.model.model_name)
    config.model.num_ctx = getattr(args, "num_ctx", config.model.num_ctx)
    if hasattr(args, "max_iterations"):
        config.runtime.max_iterations = args.max_iterations
    if hasattr(args, "artifacts_dir"):
        config.runtime.artifacts_dir = Path(args.artifacts_dir)
    if hasattr(args, "logs_dir"):
        config.runtime.logs_dir = Path(args.logs_dir)

    runtime = build_runtime(config)
    if getattr(args, "quiet", False):
        runtime.telemetry._echo_to_console = False
    else:
        print(
            (
                f"Iniciando deep research con modelo={config.model.model_name}, "
                f"num_ctx={config.model.num_ctx}, max_iterations={config.runtime.max_iterations}"
            ),
            file=sys.stderr,
            flush=True,
        )
    report = self_check_services(browser=runtime.browser, model=config.model)
    if command == "self-check":
        print(json.dumps(report.__dict__, indent=2, ensure_ascii=True))
        return 0 if report.docker_ok and report.ollama_ok and report.model_available else 1

    if not getattr(args, "skip_self_check", False):
        print("Self-check operativo completado, iniciando grafo...", file=sys.stderr, flush=True)
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
        print(json.dumps({"error": "No se pudo generar informe final"}, indent=2, ensure_ascii=True))
        return 2
    markdown_path = runtime.telemetry.write_markdown_report(final_report, label="final_report")
    final_report.markdown_artifact_path = str(markdown_path)
    checkpoint_path = runtime.telemetry.checkpoint(final_state, label="final")
    print(
        f"Informe final generado. Markdown: {markdown_path} | Checkpoint: {checkpoint_path}",
        file=sys.stderr,
        flush=True,
    )
    print(final_report.executive_answer)
    print(f"\nMarkdown report: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
