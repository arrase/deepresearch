from __future__ import annotations

from contextlib import contextmanager

import pytest
from pydantic import ValidationError

from deepresearch.config import ResearchConfig
from deepresearch.observability import langsmith_tracing


def test_langsmith_settings_load_from_config(tmp_path) -> None:
    config_root = tmp_path / "config-root"
    config = ResearchConfig.load(config_root=config_root)
    config.config_file_path.write_text(
        config.config_file_path.read_text(encoding="utf-8").replace(
            "[langsmith]\n"
            'enabled = false\n'
            "tracing = true\n"
            'endpoint = ""\n'
            'api_key = ""\n'
            'project = "DeepResearch"\n',
            "[langsmith]\n"
            "enabled = true\n"
            "tracing = true\n"
            'endpoint = "https://eu.api.smith.langchain.com"\n'
            'api_key = "test-key"\n'
            'project = "Deepresearch"\n',
        ),
        encoding="utf-8",
    )

    reloaded = ResearchConfig.load(config_root=config_root)

    assert reloaded.langsmith.enabled is True
    assert reloaded.langsmith.endpoint == "https://eu.api.smith.langchain.com"
    assert reloaded.langsmith.api_key == "test-key"
    assert reloaded.langsmith.project == "Deepresearch"


def test_langsmith_enabled_requires_api_key(tmp_path) -> None:
    config_root = tmp_path / "config-root"
    config = ResearchConfig.load(config_root=config_root)
    config.config_file_path.write_text(
        config.config_file_path.read_text(encoding="utf-8").replace(
            'enabled = false',
            "enabled = true",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        ResearchConfig.load(config_root=config_root)


def test_langsmith_tracing_noops_when_disabled(monkeypatch, tmp_path) -> None:
    config = ResearchConfig.load(config_root=tmp_path / "config-root")

    def unexpected_client(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Client should not be constructed when LangSmith is disabled")

    @contextmanager
    def passthrough(**kwargs):  # type: ignore[no-untyped-def]
        yield

    monkeypatch.setattr("deepresearch.observability.Client", unexpected_client)
    monkeypatch.setattr("deepresearch.observability.tracing_context", passthrough)

    with langsmith_tracing(config):
        pass
