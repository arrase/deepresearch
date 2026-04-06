"""CLI entrypoint for the deep research system."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path
from types import FrameType

from pydantic import ValidationError

from .config import DEFAULT_CONFIG_FILENAME, ResearchConfig, resolve_config_root
from .context_manager import ContextManager
from .core.llm import LLMWorkers
from .graph import build_graph
from .observability import configure_logging, langsmith_tracing
from .output_utils import write_markdown_report, write_pdf_report
from .runtime import ResearchRuntime
from .state import build_initial_state
from .tools import TavilySearchClient

logger = logging.getLogger(__name__)


def build_runtime(config: ResearchConfig) -> ResearchRuntime:
    return ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=LLMWorkers(config),
        search_client=TavilySearchClient(config.search),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditable deep research with LangGraph, Ollama, and Tavily")
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
        help=(
            "Logging verbosity level: 0 disables progress logs, "
            "1 shows stage progress, 2 adds decision summaries, "
            "3 adds detailed local diagnostics"
        ),
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


def _print_user_error(title: str, details: list[str]) -> None:
    print(f"Error: {title}", file=sys.stderr)
    for detail in details:
        print(f"  - {detail}", file=sys.stderr)


def _format_config_location(location: tuple[object, ...]) -> str:
    if not location:
        return "<root>"

    parts: list[str] = []
    for item in location:
        if isinstance(item, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{item}]"
            else:
                parts.append(f"[{item}]")
            continue
        parts.append(str(item))

    head, *tail = parts
    if not tail:
        return f"[{head}]"
    return f"[{head}].{'.'.join(tail)}"


def _report_config_validation_error(config_file_path: Path, error: ValidationError) -> None:
    detail_lines = [f"Configuration file: {config_file_path}"]
    for issue in error.errors():
        location = _format_config_location(tuple(issue.get("loc", ())))
        message = issue.get("msg", "Invalid value")
        if issue.get("type") == "extra_forbidden":
            detail_lines.append(f"Unsupported setting {location}. Remove or rename it.")
            continue
        detail_lines.append(f"{location}: {message}.")

    detail_lines.append(
        "If this config comes from an older release, move config.toml to a backup location "
        "and run the command again to regenerate a fresh commented config."
    )
    _print_user_error("Invalid configuration", detail_lines)


def _report_config_load_error(config_file_path: Path, error: tomllib.TOMLDecodeError | OSError) -> None:
    _print_user_error(
        "Unable to load configuration",
        [
            f"Configuration file: {config_file_path}",
            f"Reason: {error}",
            "Fix the file and run the command again.",
        ],
    )


def _report_runtime_config_error(config_file_path: Path, error: ValueError) -> None:
    message = str(error)
    if "Tavily search requires an api_key" in message:
        _print_user_error(
            "Search configuration is incomplete",
            [
                f"Configuration file: {config_file_path}",
                "Missing setting: [search].api_key",
                "Add your Tavily API key and run the command again.",
            ],
        )
        return
    raise error


@contextmanager
def _sigterm_as_keyboard_interrupt():
    previous_handler = signal.getsignal(signal.SIGTERM)

    def _raise_keyboard_interrupt(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def cli() -> int:
    args = parse_args()
    config_root = resolve_config_root(args.config_root)
    config_file_path = config_root / DEFAULT_CONFIG_FILENAME

    try:
        with _sigterm_as_keyboard_interrupt():
            try:
                config = ResearchConfig.load(config_root=config_root)
            except ValidationError as error:
                _report_config_validation_error(config_file_path, error)
                return 2
            except (tomllib.TOMLDecodeError, OSError) as error:
                _report_config_load_error(config_file_path, error)
                return 2

            apply_cli_overrides(config, args)
            configure_logging(config.runtime.verbosity)

            try:
                runtime = build_runtime(config)
            except ValueError as error:
                _report_runtime_config_error(config_file_path, error)
                return 2

            if config.runtime.verbosity >= 1:
                logger.info(
                    "Starting deep research with model=%s, num_ctx=%s, max_iterations=%s, verbosity=%s, config_root=%s",
                    config.model.model_name,
                    config.model.num_ctx,
                    config.runtime.max_iterations,
                    config.runtime.verbosity,
                    config.config_root,
                )

            initial_state = build_initial_state(args.query, max_iterations=config.runtime.max_iterations)
            graph = build_graph(runtime)
            trace_metadata = {
                "model_name": config.model.model_name,
                "search_backend": "tavily",
                "config_root": str(config.config_root),
            }
            with runtime, langsmith_tracing(config, metadata=trace_metadata):
                final_state = graph.invoke(
                    initial_state,
                    config={"run_name": "deepresearch", "tags": ["cli"], "metadata": trace_metadata},
                )

            final_report = final_state.get("final_report")
            if final_report is None:
                _print_user_error(
                    "The run finished without a final report",
                    ["Re-run with --verbosity 1 or higher to inspect stage-level progress."],
                )
                return 2

            try:
                if args.markdown:
                    write_markdown_report(final_report.markdown_report, Path(args.markdown))
                elif args.pdf:
                    write_pdf_report(final_report.markdown_report, Path(args.pdf))
                elif not args.discord:
                    write_markdown_report(final_report.markdown_report, Path("report.md"))
            except OSError as error:
                output_path = args.markdown or args.pdf or "report.md"
                _print_user_error(
                    "Unable to write report output",
                    [
                        f"Output path: {Path(output_path)}",
                        f"Reason: {error}",
                    ],
                )
                return 2

            if args.discord:
                import asyncio

                from .outputs.discord import send_discord_report

                success = asyncio.run(send_discord_report(config.discord, final_report))
                if not success:
                    logger.error("Failed to send report to Discord. Check your configuration.")

            return 0
    except KeyboardInterrupt:
        _print_user_error(
            "Execution cancelled",
            ["DeepResearch stopped cleanly after the interrupt.", "No partially written report was committed to disk."],
        )
        return 130


if __name__ == "__main__":
    raise SystemExit(cli())
