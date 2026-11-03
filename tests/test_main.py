"""Tests for the orchestrator's seen-ID handling and exit codes."""

from __future__ import annotations

import json

import main
from conftest import make_config, make_paper, make_ranked


def test_papers_to_mark_seen_retries_unadded_shortlist() -> None:
    """Shortlisted papers missing from Zotero are not marked seen."""
    added = make_ranked(title="added", score=5)
    failed = make_ranked(title="failed", score=5)
    low = make_ranked(title="low", score=2)
    out = main.papers_to_mark_seen([added, failed, low], 4, ["added"])
    assert {p.title for p in out} == {"added", "low"}


class _FailingZotero:
    def __init__(self, config) -> None:
        pass

    def existing_dois(self) -> set[str]:
        return set()

    def add_papers(self, papers):
        return []


def _run(tmp_path, monkeypatch, zotero_cls, dry_run=False):
    paper = make_paper(
        title="Prescribed burning in the United States",
        abstract="Prescribed burning and wildfire in the United States.",
        doi="10.1/new",
        openalex_id="W42",
    )
    monkeypatch.setattr(main, "fetch_papers", lambda c: [paper])
    monkeypatch.setattr(main, "ZoteroClient", zotero_cls)
    config = make_config(
        interests="prescribed burning wildfire",
        enrich=False,
        project_root=tmp_path,
        thresholds=[0.3, 0.2, 0.1, 0.05],
    )
    result = main.run_pipeline(config, dry_run=dry_run)
    state = json.loads((tmp_path / "state" / "seen_ids.json").read_text()) if (
        tmp_path / "state" / "seen_ids.json"
    ).exists() else {"dois": []}
    return result, state


def test_zotero_failure_keeps_paper_unseen(tmp_path, monkeypatch) -> None:
    """A failed Zotero write leaves the paper out of seen state."""
    result, state = _run(tmp_path, monkeypatch, _FailingZotero)
    assert result.failed_count == 1
    assert "10.1/new" not in state["dois"]


def test_zotero_success_marks_seen(tmp_path, monkeypatch) -> None:
    """A successful write records the paper as seen."""

    class Ok(_FailingZotero):
        def add_papers(self, papers):
            return [p.title for p in papers]

    result, state = _run(tmp_path, monkeypatch, Ok)
    assert result.failed_count == 0
    assert "10.1/new" in state["dois"]


def test_dry_run_leaves_state_alone(tmp_path, monkeypatch) -> None:
    """Dry runs never write seen state."""
    result, state = _run(tmp_path, monkeypatch, _FailingZotero, dry_run=True)
    assert result.failed_count == 0
    assert state["dois"] == []
