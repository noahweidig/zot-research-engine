"""Deduplication against the current batch, run history, and Zotero.

Removes duplicate papers in three stages and maintains ``state/seen_ids.json``
so papers evaluated in earlier runs are never re-processed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pipeline.models import Paper

logger = logging.getLogger(__name__)

SEEN_IDS_FILENAME = Path("state") / "seen_ids.json"


def _empty_state() -> dict[str, list[str]]:
    """Return an empty seen-ids state structure."""
    return {"dois": [], "openalex_ids": []}


def load_seen_ids(path: str | Path) -> dict[str, list[str]]:
    """Load the seen-ids state file, creating it if it does not exist.

    Args:
        path: Path to ``seen_ids.json``.

    Returns:
        A dict with ``"dois"`` and ``"openalex_ids"`` lists.
    """
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_empty_state(), indent=2), encoding="utf-8")
        return _empty_state()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s (%s); starting fresh", p, exc)
        return _empty_state()
    return {
        "dois": list(data.get("dois", [])),
        "openalex_ids": list(data.get("openalex_ids", [])),
    }


def save_seen_ids(path: str | Path, papers: list[Paper]) -> None:
    """Append every evaluated paper's identifiers to the seen-ids state file.

    All evaluated papers are recorded (not only those added to Zotero), so they
    are excluded from future runs.

    Args:
        path: Path to ``seen_ids.json``.
        papers: The papers evaluated during this run.
    """
    state = load_seen_ids(path)
    dois = set(state["dois"])
    oa_ids = set(state["openalex_ids"])

    for paper in papers:
        if paper.doi:
            dois.add(paper.doi.lower())
        if paper.openalex_id:
            oa_ids.add(paper.openalex_id)

    Path(path).write_text(
        json.dumps(
            {"dois": sorted(dois), "openalex_ids": sorted(oa_ids)}, indent=2
        ),
        encoding="utf-8",
    )
    logger.info(
        "Saved seen-ids: %d DOIs, %d OpenAlex IDs", len(dois), len(oa_ids)
    )


def _dedupe_within_batch(papers: list[Paper]) -> list[Paper]:
    """Remove intra-batch duplicates by DOI, then by OpenAlex ID.

    Args:
        papers: The batch of papers.

    Returns:
        The batch with duplicates removed, preserving first-seen order.
    """
    seen_dois: set[str] = set()
    seen_oa: set[str] = set()
    result: list[Paper] = []
    for paper in papers:
        doi_key = paper.doi.lower() if paper.doi else None
        oa_key = paper.openalex_id or None
        if doi_key and doi_key in seen_dois:
            continue
        if oa_key and oa_key in seen_oa:
            continue
        if doi_key:
            seen_dois.add(doi_key)
        if oa_key:
            seen_oa.add(oa_key)
        result.append(paper)
    return result


def _dedupe_against(
    papers: list[Paper], known_dois: set[str], known_oa_ids: set[str]
) -> list[Paper]:
    """Drop papers whose DOI or OpenAlex ID is already known.

    Args:
        papers: The batch of papers.
        known_dois: Lower-cased DOIs to exclude.
        known_oa_ids: OpenAlex IDs to exclude.

    Returns:
        The filtered batch.
    """
    result: list[Paper] = []
    for paper in papers:
        doi_key = paper.doi.lower() if paper.doi else None
        if doi_key and doi_key in known_dois:
            continue
        if paper.openalex_id and paper.openalex_id in known_oa_ids:
            continue
        result.append(paper)
    return result


def deduplicate(
    papers: list[Paper],
    seen_ids: dict[str, list[str]],
    zotero_dois: set[str] | None = None,
) -> list[Paper]:
    """Deduplicate papers across all three stages.

    Order: (1) within the current batch, (2) against run history from
    ``seen_ids.json``, (3) against DOIs already present in the Zotero library.

    Args:
        papers: The freshly fetched papers.
        seen_ids: Loaded seen-ids state.
        zotero_dois: DOIs already in the Zotero library, or ``None`` to skip
            that stage.

    Returns:
        The deduplicated list of papers.
    """
    stage1 = _dedupe_within_batch(papers)
    logger.info("Dedupe (batch): removed %d", len(papers) - len(stage1))

    history_dois = {d.lower() for d in seen_ids.get("dois", [])}
    history_oa = set(seen_ids.get("openalex_ids", []))
    stage2 = _dedupe_against(stage1, history_dois, history_oa)
    logger.info("Dedupe (history): removed %d", len(stage1) - len(stage2))

    if zotero_dois:
        zdois = {d.lower() for d in zotero_dois}
        stage3 = _dedupe_against(stage2, zdois, set())
        logger.info("Dedupe (Zotero): removed %d", len(stage2) - len(stage3))
    else:
        stage3 = stage2

    logger.info("Dedupe: %d unique papers remain", len(stage3))
    return stage3
