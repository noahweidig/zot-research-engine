"""Shared data models for the pipeline.

Kept in a dedicated module so every stage can import :class:`Paper` and
:class:`RankedPaper` without creating circular imports between the fetch,
rank, and storage modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Paper:
    """A normalized research paper from any source.

    Attributes:
        title: Paper title.
        abstract: Abstract text, or ``None`` if unavailable.
        doi: Digital Object Identifier (bare, e.g. ``10.1234/abc``), or ``None``.
        authors: Author display names, in order.
        year: Publication year, or ``None``.
        journal: Journal or venue name, or ``None``.
        openalex_id: OpenAlex work id (e.g. ``W123``), or a source-prefixed id
            for papers that did not originate from OpenAlex.
        citation_count: Number of citations reported by the source.
        url: Canonical URL for the paper.
        source: Originating source, e.g. ``"openalex"``, ``"arxiv"``.
        raw: The original API response, retained for debugging.
    """

    title: str
    abstract: str | None
    doi: str | None
    authors: list[str]
    year: int | None
    journal: str | None
    openalex_id: str
    citation_count: int
    url: str
    source: str
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class RankedPaper(Paper):
    """A :class:`Paper` augmented with Gemini relevance ranking.

    Attributes:
        score: Relevance score from 1 (ignore) to 5 (must read).
        reason: Short justification for the score.
        tags: Topic tags suggested by the model.
        summary: 2-3 sentence plain-language summary.
    """

    score: int = 0
    reason: str = ""
    tags: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_paper(
        cls,
        paper: Paper,
        *,
        score: int,
        reason: str,
        tags: list[str],
        summary: str,
    ) -> "RankedPaper":
        """Build a :class:`RankedPaper` from a :class:`Paper` and ranking fields.

        Args:
            paper: The base paper.
            score: Relevance score (1-5).
            reason: Justification for the score.
            tags: Topic tags.
            summary: Plain-language summary.

        Returns:
            A new :class:`RankedPaper`.
        """
        return cls(
            title=paper.title,
            abstract=paper.abstract,
            doi=paper.doi,
            authors=paper.authors,
            year=paper.year,
            journal=paper.journal,
            openalex_id=paper.openalex_id,
            citation_count=paper.citation_count,
            url=paper.url,
            source=paper.source,
            raw=paper.raw,
            score=score,
            reason=reason,
            tags=tags,
            summary=summary,
        )
