"""Optional Crossref search source.

Queries the Crossref REST API (https://api.crossref.org/works) for recently
published works matching the configured queries. Enabled via
``sources.enable_crossref`` in config.yaml. Requires no API key; sends
``OPENALEX_EMAIL`` in the polite ``mailto`` parameter.
"""

from __future__ import annotations

import datetime as dt
import logging
import re

import requests

from config import PipelineConfig
from pipeline.http import http_retry
from pipeline.models import Paper

logger = logging.getLogger(__name__)

CROSSREF_BASE_URL = "https://api.crossref.org/works"
_JATS_TAG = re.compile(r"<[^>]+>")


@http_retry
def _get(params: dict, mailto: str) -> dict:
    """Perform a GET request against Crossref and return parsed JSON.

    Args:
        params: Query parameters.
        mailto: Email for the Crossref polite pool.

    Returns:
        The parsed JSON response body.
    """
    resp = requests.get(
        CROSSREF_BASE_URL,
        params={**params, "mailto": mailto},
        headers={"User-Agent": f"zot-research-engine (mailto:{mailto})"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _clean_abstract(abstract: str | None) -> str | None:
    """Strip JATS/XML tags from a Crossref abstract.

    Args:
        abstract: The raw abstract markup, or ``None``.

    Returns:
        Plain-text abstract, or ``None`` if empty.
    """
    if not abstract:
        return None
    text = _JATS_TAG.sub("", abstract).strip()
    return text or None


def _normalize(item: dict) -> Paper | None:
    """Normalize a single Crossref work into a :class:`Paper`.

    Args:
        item: A work object from the Crossref API.

    Returns:
        A :class:`Paper`, or ``None`` if it has no abstract.
    """
    title_list = item.get("title") or []
    title = title_list[0] if title_list else "(untitled)"
    abstract = _clean_abstract(item.get("abstract"))
    if not abstract:
        logger.warning("Skipping Crossref paper with no abstract: %s", title)
        return None

    authors = [
        " ".join(filter(None, [a.get("given"), a.get("family")]))
        for a in item.get("author", [])
    ]
    authors = [a for a in authors if a]

    container = item.get("container-title") or []
    journal = container[0] if container else None

    date_parts = (item.get("issued") or {}).get("date-parts") or [[None]]
    year = date_parts[0][0] if date_parts and date_parts[0] else None

    doi = item.get("DOI")

    return Paper(
        title=title,
        abstract=abstract,
        doi=doi.strip() if doi else None,
        authors=authors,
        year=int(year) if year else None,
        journal=journal,
        openalex_id=f"crossref:{doi}" if doi else f"crossref:{item.get('URL', '')}",
        citation_count=int(item.get("is-referenced-by-count", 0) or 0),
        url=item.get("URL", ""),
        source="crossref",
        raw=item,
    )


def fetch_crossref(config: PipelineConfig) -> list[Paper]:
    """Fetch recently published works from Crossref for all configured queries.

    Args:
        config: The pipeline configuration.

    Returns:
        A list of :class:`Paper` objects.
    """
    if not config.sources.enable_crossref:
        return []

    from_date = (
        dt.date.today() - dt.timedelta(days=config.search.days_back)
    ).isoformat()
    papers: list[Paper] = []

    for query in config.search.queries:
        try:
            data = _get(
                {
                    "query": query,
                    "filter": f"from-pub-date:{from_date}",
                    "rows": config.search.max_results_per_query,
                    "select": (
                        "title,abstract,DOI,author,container-title,"
                        "issued,URL,is-referenced-by-count"
                    ),
                    "sort": "published",
                    "order": "desc",
                },
                config.secrets.openalex_email,
            )
        except requests.exceptions.RequestException as exc:
            logger.error("Crossref query failed for %r: %s", query, exc)
            continue

        found = [
            paper
            for item in data.get("message", {}).get("items", [])
            if (paper := _normalize(item)) is not None
        ]
        logger.info("Crossref: %d results for query %r", len(found), query)
        papers.extend(found)

    return papers
