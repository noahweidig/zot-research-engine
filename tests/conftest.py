"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    GeminiConfig,
    PipelineConfig,
    PipelineSettings,
    RankerConfig,
    RelevanceConfig,
    SearchConfig,
    Secrets,
    SourcesConfig,
    ZoteroConfig,
)
from pipeline.models import Paper, RankedPaper  # noqa: E402


def make_config(
    *,
    min_score: int = 4,
    concurrency: int = 2,
    interests: str = "GIS wildfire",
    thresholds: list[float] | None = None,
    enrich: bool = True,
) -> PipelineConfig:
    """Build a minimal in-memory :class:`PipelineConfig` for tests.

    Args:
        min_score: Minimum score threshold.
        concurrency: Gemini max concurrency.
        interests: Researcher interests string.
        thresholds: Local-ranker coverage cutoffs (defaults to the shipped set).
        enrich: Whether Gemini enrichment is enabled.

    Returns:
        A fully populated configuration.
    """
    return PipelineConfig(
        search=SearchConfig(queries=["x"], days_back=7, max_results_per_query=10),
        relevance=RelevanceConfig(min_score=min_score, interests=interests),
        zotero=ZoteroConfig(library_type="user", inbox_collection="AI Inbox"),
        sources=SourcesConfig(
            enable_semantic_scholar=False,
            enable_arxiv=False,
            enable_crossref=False,
        ),
        pipeline=PipelineSettings(
            log_to_file=False, log_file="x.log", gemini_max_concurrency=concurrency
        ),
        gemini=GeminiConfig(model="gemini-2.0-flash", temperature=0.0, enrich=enrich),
        ranker=RankerConfig(thresholds=thresholds or [0.55, 0.4, 0.25, 0.1]),
        secrets=Secrets(
            gemini_api_key="k",
            zotero_api_key="k",
            zotero_user_id="1",
            zotero_library_type="user",
            openalex_email="a@b.c",
        ),
        project_root=Path("."),
    )


def make_ranked(
    *,
    title: str = "A Paper",
    abstract: str | None = "An abstract.",
    score: int = 5,
    reason: str = "r",
    tags: list[str] | None = None,
    summary: str = "s",
) -> RankedPaper:
    """Construct a :class:`RankedPaper` with sensible defaults for tests."""
    return RankedPaper.from_paper(
        make_paper(title=title, abstract=abstract),
        score=score,
        reason=reason,
        tags=tags if tags is not None else ["t"],
        summary=summary,
    )


def make_paper(
    *,
    title: str = "A Paper",
    doi: str | None = "10.1/abc",
    openalex_id: str = "W1",
    abstract: str | None = "An abstract.",
    source: str = "openalex",
) -> Paper:
    """Construct a :class:`Paper` with sensible defaults for tests.

    Args:
        title: Paper title.
        doi: DOI, or ``None``.
        openalex_id: OpenAlex id.
        abstract: Abstract text, or ``None``.
        source: Originating source.

    Returns:
        A populated :class:`Paper`.
    """
    return Paper(
        title=title,
        abstract=abstract,
        doi=doi,
        authors=["Jane Doe"],
        year=2025,
        journal="Journal",
        openalex_id=openalex_id,
        citation_count=0,
        url="https://example.org",
        source=source,
        raw={},
    )
