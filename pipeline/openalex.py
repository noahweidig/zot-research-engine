"""OpenAlex search source.

Queries the OpenAlex REST API (https://api.openalex.org/works) for recent works
matching the configured queries, normalizes each result into a :class:`Paper`,
and skips any result without an abstract.
"""

from __future__ import annotations

import datetime as dt
import logging

import requests

from config import PipelineConfig
from pipeline.http import http_retry
from pipeline.models import Paper

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org/works"
# OpenAlex caps per_page at 200.
_MAX_PER_PAGE = 200


@http_retry
def _get(url: str, params: dict) -> dict:
    """Perform a GET request against OpenAlex and return parsed JSON.

    Args:
        url: The request URL.
        params: Query parameters.

    Returns:
        The parsed JSON response body.
    """
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _reconstruct_abstract(inverted_index: dict | None) -> str | None:
    """Reconstruct an abstract from an OpenAlex inverted index.

    Args:
        inverted_index: The ``abstract_inverted_index`` field, or ``None``.

    Returns:
        The reconstructed abstract text, or ``None`` if unavailable.
    """
    if not inverted_index:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for i in idxs:
            positions.append((i, word))
    if not positions:
        return None
    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


def _normalize(work: dict) -> Paper | None:
    """Normalize a single OpenAlex work into a :class:`Paper`.

    Args:
        work: A work object from the OpenAlex API.

    Returns:
        A :class:`Paper`, or ``None`` if the work has no abstract.
    """
    title = work.get("title") or work.get("display_name") or "(untitled)"
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    if not abstract:
        logger.warning("Skipping paper with no abstract: %s", title)
        return None

    doi = work.get("doi")
    if doi:
        doi = doi.replace("https://doi.org/", "").strip() or None

    authors = [
        a.get("author", {}).get("display_name", "")
        for a in work.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]

    venue = (work.get("primary_location") or {}).get("source") or {}
    journal = venue.get("display_name")

    openalex_id = (work.get("id") or "").replace("https://openalex.org/", "")

    return Paper(
        title=title,
        abstract=abstract,
        doi=doi,
        authors=authors,
        year=work.get("publication_year"),
        journal=journal,
        openalex_id=openalex_id,
        citation_count=int(work.get("cited_by_count", 0) or 0),
        url=work.get("id", ""),
        source="openalex",
        raw=work,
    )


def fetch_openalex(config: PipelineConfig) -> list[Paper]:
    """Fetch recent papers from OpenAlex for all configured queries.

    Args:
        config: The pipeline configuration.

    Returns:
        A list of :class:`Paper` objects (those lacking abstracts are skipped).
    """
    from_date = (
        dt.date.today() - dt.timedelta(days=config.search.days_back)
    ).isoformat()
    papers: list[Paper] = []

    for query in config.search.queries:
        try:
            found = _fetch_query(query, from_date, config)
        except requests.exceptions.RequestException as exc:
            logger.error("OpenAlex query failed for %r: %s", query, exc)
            continue
        logger.info("OpenAlex: %d results for query %r", len(found), query)
        papers.extend(found)

    return papers


def _fetch_query(query: str, from_date: str, config: PipelineConfig) -> list[Paper]:
    """Fetch and paginate results for a single query.

    Args:
        query: The search string.
        from_date: ISO date lower bound for publication date.
        config: The pipeline configuration.

    Returns:
        Normalized papers for this query.
    """
    per_page = min(config.search.max_results_per_query, _MAX_PER_PAGE)
    results: list[Paper] = []
    cursor = "*"

    while len(results) < config.search.max_results_per_query:
        data = _get(
            OPENALEX_BASE_URL,
            {
                "search": query,
                "filter": f"from_publication_date:{from_date}",
                "per_page": per_page,
                "cursor": cursor,
                "mailto": config.secrets.openalex_email,
            },
        )
        works = data.get("results", [])
        for work in works:
            paper = _normalize(work)
            if paper is not None:
                results.append(paper)
            if len(results) >= config.search.max_results_per_query:
                break

        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor or not works:
            break

    return results
