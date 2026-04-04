"""CLI entrypoint for the deep research system."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import ResearchConfig
from .context_manager import ContextManager
from .core.llm import LLMWorkers
from .graph import build_graph
from .observability import configure_logging, langsmith_tracing
from .output_utils import generate_pdf
from .runtime import ResearchRuntime, SearchClientLike
from .state import build_initial_state
from .tools import DuckDuckGoSearchClient, LightpandaDockerManager, TavilySearchClient

logger = logging.getLogger(__name__)


def build_runtime(config: ResearchConfig) -> ResearchRuntime:
    search_client: SearchClientLike
    if config.search.backend != "tavily":
        raise ValueError(f"Unsupported search backend: {config.search.backend}")
    search_client = TavilySearchClient(config.search)

    fallback_client: SearchClientLike | None = None
    if config.search.fallback_backend == "duckduckgo":
        fallback_client = DuckDuckGoSearchClient(config.search)

    return ResearchRuntime(
        config=config,
        context_manager=ContextManager(config),
        llm_workers=LLMWorkers(config),
        browser=LightpandaDockerManager(config.browser),
        search_client=search_client,
        fallback_search_client=fallback_client,
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


def cli() -> int:
    args = parse_args()
    config = ResearchConfig.load(config_root=args.config_root)
    apply_cli_overrides(config, args)
    configure_logging(config.runtime.verbosity)

    runtime = build_runtime(config)
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
        "search_backend": config.search.backend,
        "config_root": str(config.config_root),
    }
    with langsmith_tracing(config, metadata=trace_metadata):
        final_state = graph.invoke(
            initial_state,
            config={"run_name": "deepresearch", "tags": ["cli"], "metadata": trace_metadata},
        )
    final_report = final_state.get("final_report")
    if final_report is None:
        print(json.dumps({"error": "Failed to generate final report"}, indent=2, ensure_ascii=True))
        return 2

    if args.markdown:
        output_path = Path(args.markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(final_report.markdown_report, encoding="utf-8")
    elif args.pdf:
        output_path = Path(args.pdf)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        generate_pdf(final_report.markdown_report, output_path)
    elif not args.discord:
        output_path = Path("report.md")
        output_path.write_text(final_report.markdown_report, encoding="utf-8")

    if args.discord:
        import asyncio

        from .outputs.discord import send_discord_report

        success = asyncio.run(send_discord_report(config.discord, final_report))
        if not success:
            logger.error("Failed to send report to Discord. Check your configuration.")

    print(final_report.executive_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
