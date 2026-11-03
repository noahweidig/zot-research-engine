"""Tests for report retention pruning."""

from __future__ import annotations

import datetime as dt

from conftest import make_config
from pipeline.report import prune_reports


def test_prune_reports_deletes_only_old_dated_files(tmp_path) -> None:
    """Old dated .md/.json files go; recent and non-dated files stay."""
    reports = tmp_path / "reports"
    reports.mkdir()
    for name in ["2026-01-01.md", "2026-01-01.json", "2026-06-01.md", "notes.md"]:
        (reports / name).write_text("x")
    cfg = make_config(project_root=tmp_path, report_retention_days=30)
    assert prune_reports(cfg, today=dt.date(2026, 6, 10)) == 2
    assert sorted(p.name for p in reports.iterdir()) == ["2026-06-01.md", "notes.md"]
