from __future__ import annotations

import pytest

from deepresearch.output_utils import write_markdown_report


def test_write_markdown_report_preserves_previous_file_on_replace_failure(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "report.md"
    output_path.write_text("existing report", encoding="utf-8")

    def fail_replace(source: str, destination: str) -> None:
        del source, destination
        raise OSError("disk full")

    monkeypatch.setattr("deepresearch.output_utils.os.replace", fail_replace)

    with pytest.raises(OSError, match="disk full"):
        write_markdown_report("new report", output_path)

    assert output_path.read_text(encoding="utf-8") == "existing report"
    assert list(tmp_path.glob(".report.md.*.tmp")) == []
