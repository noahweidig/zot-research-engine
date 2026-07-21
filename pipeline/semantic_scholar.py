"""Semantic Scholar abstract enrichment.

Used only to fill in missing abstracts (via DOI) for papers already fetched
from other sources — not as an independent search source.
"""

from __future__ import annotations

import logging

import requests

from config import PipelineConfig
from pipeline.http import http_retry
from pipeline.models import Paper

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"


@http_retry
def _get_abstract(doi: str) -> str | None:
    """Fetch a paper's abstract from Semantic Scholar by DOI.

    Args:
        doi: The bare DOI (e.g. ``10.1234/abc``).

    Returns:
        The abstract text, or ``None`` if not found.
    """
    url = f"{SEMANTIC_SCHOLAR_BASE_URL}/DOI:{doi}"
    resp = requests.get(url, params={"fields": "abstract"}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("abstract")


def enrich_abstracts(papers: list[Paper], config: PipelineConfig) -> int:
    """Fill in missing abstracts from Semantic Scholar where possible.

    Papers are mutated in place; only those missing an abstract and having a DOI
    are queried.

    Args:
        papers: The papers to enrich.
        config: The pipeline configuration.

    Returns:
        The number of abstracts successfully enriched.
    """
    if not config.sources.enable_semantic_scholar:
        return 0

    enriched = 0
    for paper in papers:
        if paper.abstract or not paper.doi:
            continue
        try:
            abstract = _get_abstract(paper.doi)
        except requests.exceptions.RequestException as exc:
            logger.warning(
                "Semantic Scholar enrichment failed for DOI %s: %s", paper.doi, exc
            )
            continue
        if abstract:
            paper.abstract = abstract
            enriched += 1
            logger.info("Enriched abstract for: %s", paper.title)

    logger.info("Semantic Scholar: enriched %d abstract(s)", enriched)
    return enriched
