"""Orchestrator for the zot-research-engine pipeline.

Contains no business logic — it only wires together the pipeline stages defined
in the :mod:`pipeline` package and handles CLI flags.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys

from config import ConfigError, PipelineConfig, load_config
from pipeline import arxiv, crossref, openalex, semantic_scholar
from pipeline.deduplicate import (
    SEEN_IDS_FILENAME,
    deduplicate,
    load_seen_ids,
    save_seen_ids,
)
from pipeline.gemini_ranker import enrich_papers
from pipeline.local_ranker import rank_papers
from pipeline.models import Paper
from pipeline.report import (
    PipelineResult,
    generate_report,
    latest_report_path,
    load_result_from_json,
)
from pipeline.zotero_client import ZoteroClient

logger = logging.getLogger(__name__)


def _configure_logging(config: PipelineConfig) -> None:
    """Configure root logging to stdout and, optionally, a file.

    Args:
        config: The pipeline configuration.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if config.pipeline.log_to_file:
        log_path = config.project_root / config.pipeline.log_file
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def fetch_papers(config: PipelineConfig) -> list[Paper]:
    """Fetch papers from OpenAlex plus any enabled optional sources.

    Args:
        config: The pipeline configuration.

    Returns:
        The combined list of fetched papers.
    """
    papers: list[Paper] = []
    papers.extend(openalex.fetch_openalex(config))
    if config.sources.enable_arxiv:
        papers.extend(arxiv.fetch_arxiv(config))
    if config.sources.enable_crossref:
        papers.extend(crossref.fetch_crossref(config))

    semantic_scholar.enrich_abstracts(papers, config)
    papers = [p for p in papers if p.abstract]

    logger.info("Fetched %d paper(s) with abstracts total", len(papers))
    return papers


def run_pipeline(config: PipelineConfig, dry_run: bool = False) -> PipelineResult:
    """Run the full discovery pipeline end to end.

    Args:
        config: The pipeline configuration.
        dry_run: If ``True``, run everything but skip writing to Zotero.

    Returns:
        A :class:`PipelineResult` describing the run.
    """
    seen_path = config.project_root / SEEN_IDS_FILENAME
    seen_ids = load_seen_ids(seen_path)

    papers = fetch_papers(config)
    fetched = len(papers)

    zotero_client: ZoteroClient | None = None
    zotero_dois: set[str] = set()
    try:
        zotero_client = ZoteroClient(config)
        zotero_dois = zotero_client.existing_dois()
    except Exception as exc:  # noqa: BLE001 - Zotero must not abort the run
        logger.error("Zotero unavailable; continuing without it: %s", exc)

    unique = deduplicate(papers, seen_ids, zotero_dois)
    duplicates_removed = fetched - len(unique)

    # Score every candidate for free, offline — this never depends on Gemini
    # quota, so a run always produces results.
    ranked = rank_papers(unique, config)

    added_titles: list[str] = []
    to_add = [p for p in ranked if p.score >= config.relevance.min_score]

    # Reserve Gemini for the small shortlist that will actually be filed:
    # polish its summaries/reasons/tags, best-effort. Failure is non-fatal.
    if to_add and config.gemini.enrich:
        enrich_papers(to_add, config)

    if dry_run:
        logger.info("Dry run: skipping Zotero writes (%d would be added)", len(to_add))
    elif zotero_client is not None and to_add:
        added_titles = zotero_client.add_papers(to_add)

    # Record every evaluated paper so future runs skip them.
    save_seen_ids(seen_path, unique)

    result = PipelineResult(
        date=dt.date.today().isoformat(),
        fetched=fetched,
        duplicates_removed=duplicates_removed,
        evaluated=len(ranked),
        ranked=ranked,
        added_titles=added_titles,
        min_score=config.relevance.min_score,
        dry_run=dry_run,
    )
    return generate_report(result, config)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Automated AI research discovery pipeline for Zotero."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full pipeline but skip writing to Zotero.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate the last report without fetching or ranking.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=None,
        help="Override max_results_per_query from config.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=None,
        help="Override min_score from config.",
    )
    return parser.parse_args(argv)


def _apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    """Apply CLI overrides to the loaded configuration.

    Args:
        config: The loaded configuration.
        args: Parsed CLI arguments.

    Returns:
        A new configuration with overrides applied.
    """
    import dataclasses

    search = config.search
    relevance = config.relevance
    if args.max_results is not None:
        search = dataclasses.replace(search, max_results_per_query=args.max_results)
    if args.min_score is not None:
        relevance = dataclasses.replace(relevance, min_score=args.min_score)
    return dataclasses.replace(config, search=search, relevance=relevance)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv``).

    Returns:
        Process exit code (0 on success, 1 on configuration error).
    """
    args = _parse_args(argv)
    try:
        config = load_config()
    except ConfigError as exc:
        logging.basicConfig(level=logging.INFO)
        logger.error("Configuration error: %s", exc)
        return 1

    config = _apply_overrides(config, args)
    _configure_logging(config)

    if args.report_only:
        path = latest_report_path(config)
        if path is None:
            logger.error("No previous report found to regenerate.")
            return 1
        logger.info("Regenerating report from %s", path)
        result = load_result_from_json(path)
        generate_report(result, config)
        return 0

    result = run_pipeline(config, dry_run=args.dry_run)
    logger.info(
        "Done. Fetched=%d Evaluated=%d Added=%d Report=%s",
        result.fetched,
        result.evaluated,
        result.added_count,
        result.markdown_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
