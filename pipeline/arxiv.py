"""Optional arXiv search source.

Queries the arXiv Atom API (http://export.arxiv.org/api/query) for recent
preprints matching the configured queries. Enabled via
``sources.enable_arxiv`` in config.yaml. Requires no API key.
"""

from __future__ import annotations

import datetime as dt
import logging
import xml.etree.ElementTree as ET

import requests

from config import PipelineConfig
from pipeline.http import http_retry
from pipeline.models import Paper

logger = logging.getLogger(__name__)

ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV_NS = "{http://arxiv.org/schemas/atom}"


@http_retry
def _get(params: dict) -> str:
    """Fetch raw Atom XML from the arXiv API.

    Args:
        params: Query parameters.

    Returns:
        The response body as text.
    """
    resp = requests.get(ARXIV_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_entry(entry: ET.Element, cutoff: dt.date) -> Paper | None:
    """Parse a single arXiv Atom ``entry`` element into a :class:`Paper`.

    Args:
        entry: The Atom entry element.
        cutoff: Papers published before this date are skipped.

    Returns:
        A :class:`Paper`, or ``None`` if it is too old or lacks an abstract.
    """
    published = entry.findtext(f"{_ATOM}published") or ""
    try:
        pub_date = dt.datetime.fromisoformat(published.replace("Z", "+00:00")).date()
    except ValueError:
        pub_date = None
    if pub_date is not None and pub_date < cutoff:
        return None

    title = (entry.findtext(f"{_ATOM}title") or "(untitled)").strip()
    abstract = (entry.findtext(f"{_ATOM}summary") or "").strip() or None
    if not abstract:
        logger.warning("Skipping arXiv paper with no abstract: %s", title)
        return None

    authors = [
        (a.findtext(f"{_ATOM}name") or "").strip()
        for a in entry.findall(f"{_ATOM}author")
        if (a.findtext(f"{_ATOM}name") or "").strip()
    ]

    arxiv_url = entry.findtext(f"{_ATOM}id") or ""
    doi = entry.findtext(f"{_ARXIV_NS}doi")

    return Paper(
        title=title,
        abstract=abstract,
        doi=doi.strip() if doi else None,
        authors=authors,
        year=pub_date.year if pub_date else None,
        journal="arXiv",
        openalex_id=f"arxiv:{arxiv_url.rsplit('/', 1)[-1]}",
        citation_count=0,
        url=arxiv_url,
        source="arxiv",
        raw={"published": published},
    )


def fetch_arxiv(config: PipelineConfig) -> list[Paper]:
    """Fetch recent preprints from arXiv for all configured queries.

    Args:
        config: The pipeline configuration.

    Returns:
        A list of :class:`Paper` objects.
    """
    if not config.sources.enable_arxiv:
        return []

    cutoff = dt.date.today() - dt.timedelta(days=config.search.days_back)
    papers: list[Paper] = []

    for query in config.search.queries:
        try:
            xml = _get(
                {
                    "search_query": f"all:{query}",
                    "start": 0,
                    "max_results": config.search.max_results_per_query,
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
            )
            root = ET.fromstring(xml)
        except (requests.exceptions.RequestException, ET.ParseError) as exc:
            logger.error("arXiv query failed for %r: %s", query, exc)
            continue

        found = [
            paper
            for entry in root.findall(f"{_ATOM}entry")
            if (paper := _parse_entry(entry, cutoff)) is not None
        ]
        logger.info("arXiv: %d results for query %r", len(found), query)
        papers.extend(found)

    return papers
