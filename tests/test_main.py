from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import cast
from unittest.mock import MagicMock

from deepresearch.config import ResearchConfig
from deepresearch.main import cli
from deepresearch.runtime import LLMWorkersLike, ResearchRuntime


@contextmanager
def passthrough_tracing(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args, kwargs
    yield


def make_args(*, config_root: str) -> argparse.Namespace:
    return argparse.Namespace(
        query="What changed?",
        markdown=None,
        pdf=None,
        config_root=config_root,
        model=None,
        num_ctx=None,
        max_iterations=None,
        verbosity=None,
        discord=False,
    )


def test_cli_reports_missing_search_api_key(tmp_path, monkeypatch, capsys) -> None:
    config_root = tmp_path / "config-root"
    args = make_args(config_root=str(config_root))

    monkeypatch.setattr("deepresearch.main.parse_args", lambda: args)
    monkeypatch.setattr("deepresearch.main.configure_logging", lambda verbosity: None)

    exit_code = cli()
    captured = capsys.readouterr()

    assert exit_code == 2
    assert str((config_root / "config.toml").resolve()) in captured.err
    assert "Missing setting: [search].api_key" in captured.err


def test_cli_reports_unknown_config_field_and_recovery_hint(tmp_path, monkeypatch, capsys) -> None:
    config_root = tmp_path / "config-root"
    config = ResearchConfig.load(config_root=config_root)
    config.config_file_path.write_text(
        config.config_file_path.read_text(encoding="utf-8").replace(
            'language = "English" # Language used for the final report.',
            'language = "English" # Language used for the final report.\nunknown_runtime_field = 99',
        ),
        encoding="utf-8",
    )
    args = make_args(config_root=str(config_root))

    monkeypatch.setattr("deepresearch.main.parse_args", lambda: args)
    monkeypatch.setattr("deepresearch.main.configure_logging", lambda verbosity: None)

    exit_code = cli()
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Unsupported setting [runtime].unknown_runtime_field" in captured.err
    assert "backup location" in captured.err


def test_cli_returns_130_and_closes_runtime_on_keyboard_interrupt(tmp_path, monkeypatch, capsys) -> None:
    config_root = tmp_path / "config-root"
    ResearchConfig.load(config_root=config_root)
    args = make_args(config_root=str(config_root))
    search_client = MagicMock()
    runtime = ResearchRuntime(
        config=ResearchConfig(),
        context_manager=MagicMock(),
        llm_workers=cast(LLMWorkersLike, object()),
        search_client=search_client,
    )

    class InterruptingGraph:
        def invoke(self, state: object, config: object) -> object:
            del state, config
            raise KeyboardInterrupt()

    monkeypatch.setattr("deepresearch.main.parse_args", lambda: args)
    monkeypatch.setattr("deepresearch.main.configure_logging", lambda verbosity: None)
    monkeypatch.setattr("deepresearch.main.build_runtime", lambda config: runtime)
    monkeypatch.setattr("deepresearch.main.build_graph", lambda runtime: InterruptingGraph())
    monkeypatch.setattr("deepresearch.main.langsmith_tracing", passthrough_tracing)

    exit_code = cli()
    captured = capsys.readouterr()

    assert exit_code == 130
    assert search_client.close.call_count == 1
    assert "Execution cancelled" in captured.err
