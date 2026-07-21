"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when running `pytest` from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.models import Paper  # noqa: E402


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
