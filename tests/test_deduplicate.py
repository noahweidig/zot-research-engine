"""Unit tests for pipeline.deduplicate. All external I/O is mocked or tmp-based."""

from __future__ import annotations

import json

from conftest import make_paper

from pipeline.deduplicate import (
    deduplicate,
    load_seen_ids,
    save_seen_ids,
)


def test_dedupe_by_doi_within_batch() -> None:
    """Two papers sharing a DOI collapse to one."""
    papers = [
        make_paper(doi="10.1/x", openalex_id="W1"),
        make_paper(doi="10.1/x", openalex_id="W2"),
    ]
    result = deduplicate(papers, {"dois": [], "openalex_ids": []})
    assert len(result) == 1


def test_dedupe_by_doi_is_case_insensitive() -> None:
    """DOI matching ignores case."""
    papers = [
        make_paper(doi="10.1/ABC", openalex_id="W1"),
        make_paper(doi="10.1/abc", openalex_id="W2"),
    ]
    result = deduplicate(papers, {"dois": [], "openalex_ids": []})
    assert len(result) == 1


def test_dedupe_by_openalex_id_within_batch() -> None:
    """Two papers sharing an OpenAlex id (no DOI) collapse to one."""
    papers = [
        make_paper(doi=None, openalex_id="W9"),
        make_paper(doi=None, openalex_id="W9"),
    ]
    result = deduplicate(papers, {"dois": [], "openalex_ids": []})
    assert len(result) == 1


def test_dedupe_against_history() -> None:
    """Papers whose id appears in seen_ids history are dropped."""
    papers = [
        make_paper(doi="10.1/new", openalex_id="W1"),
        make_paper(doi="10.1/old", openalex_id="W2"),
    ]
    seen = {"dois": ["10.1/old"], "openalex_ids": []}
    result = deduplicate(papers, seen)
    assert [p.doi for p in result] == ["10.1/new"]


def test_papers_already_in_zotero_are_skipped() -> None:
    """Papers whose DOI is already in the Zotero library are skipped."""
    papers = [
        make_paper(doi="10.1/keep", openalex_id="W1"),
        make_paper(doi="10.1/inzotero", openalex_id="W2"),
    ]
    zotero_dois = {"10.1/inzotero"}
    result = deduplicate(papers, {"dois": [], "openalex_ids": []}, zotero_dois)
    assert [p.doi for p in result] == ["10.1/keep"]


def test_load_seen_ids_creates_missing_file(tmp_path) -> None:
    """Loading a non-existent state file creates it with empty lists."""
    path = tmp_path / "state" / "seen_ids.json"
    state = load_seen_ids(path)
    assert state == {"dois": [], "openalex_ids": []}
    assert path.exists()


def test_save_seen_ids_appends_all_evaluated(tmp_path) -> None:
    """Saving appends DOIs and OpenAlex ids for every evaluated paper."""
    path = tmp_path / "seen_ids.json"
    path.write_text(json.dumps({"dois": ["10.1/pre"], "openalex_ids": ["W0"]}))
    papers = [
        make_paper(doi="10.1/a", openalex_id="W1"),
        make_paper(doi=None, openalex_id="W2"),
    ]
    save_seen_ids(path, papers)
    saved = json.loads(path.read_text())
    assert "10.1/pre" in saved["dois"]
    assert "10.1/a" in saved["dois"]
    assert set(saved["openalex_ids"]) == {"W0", "W1", "W2"}


def test_save_then_load_roundtrip_dedupes_next_run(tmp_path) -> None:
    """Ids saved after a run exclude the same papers on the next run."""
    path = tmp_path / "seen_ids.json"
    batch = [make_paper(doi="10.1/z", openalex_id="W7")]
    save_seen_ids(path, batch)
    seen = load_seen_ids(path)
    result = deduplicate(batch, seen)
    assert result == []
