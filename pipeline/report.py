"""Report generation.

Writes a human-readable Markdown report and a machine-readable JSON report to
``reports/YYYY-MM-DD.md`` and ``reports/YYYY-MM-DD.json``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from config import PipelineConfig
from pipeline.models import RankedPaper

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Aggregate result of a pipeline run, used for reporting.

    Attributes:
        date: The run date (ISO).
        fetched: Total papers fetched across all sources.
        duplicates_removed: Number of papers removed during deduplication.
        evaluated: Number of candidate papers scored by Gemini.
        ranked: All ranked papers (sorted by descending score).
        added_titles: Titles successfully added to Zotero.
        min_score: The minimum score threshold used for this run.
        dry_run: Whether Zotero writes were skipped.
        markdown_path: Path to the written Markdown report, if any.
        json_path: Path to the written JSON report, if any.
    """

    date: str
    fetched: int = 0
    duplicates_removed: int = 0
    evaluated: int = 0
    ranked: list[RankedPaper] = field(default_factory=list)
    added_titles: list[str] = field(default_factory=list)
    min_score: int = 4
    dry_run: bool = False
    markdown_path: str | None = None
    json_path: str | None = None

    @property
    def added_count(self) -> int:
        """Number of papers added to Zotero."""
        return len(self.added_titles)

    @property
    def ignored(self) -> list[RankedPaper]:
        """Ranked papers that fell below the score threshold."""
        return [p for p in self.ranked if p.score < self.min_score]

    @property
    def average_score(self) -> float:
        """Mean Gemini score across evaluated papers (0.0 if none)."""
        if not self.ranked:
            return 0.0
        return sum(p.score for p in self.ranked) / len(self.ranked)


def _format_authors(authors: list[str]) -> str:
    """Format an author list as ``Last, F.`` entries.

    Args:
        authors: Author display names.

    Returns:
        A comma-separated formatted author string.
    """
    formatted = []
    for name in authors:
        parts = name.split()
        if len(parts) >= 2:
            formatted.append(f"{parts[-1]}, {parts[0][0]}.")
        elif parts:
            formatted.append(parts[0])
    return ", ".join(formatted) if formatted else "Unknown"


def _render_markdown(result: PipelineResult) -> str:
    """Render the Markdown report body.

    Args:
        result: The pipeline result.

    Returns:
        The Markdown document as a string.
    """
    added = [p for p in result.ranked if p.title in set(result.added_titles)]
    lines: list[str] = [
        f"# Daily Literature Review — {result.date}",
        "",
        "## Summary Statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Papers fetched | {result.fetched} |",
        f"| Duplicates removed | {result.duplicates_removed} |",
        f"| Candidates evaluated | {result.evaluated} |",
        f"| Added to Zotero | {result.added_count} |",
        f"| Ignored | {len(result.ignored)} |",
        f"| Average Gemini score | {result.average_score:.1f} |",
        "",
        "---",
        "",
        "## Added to Zotero",
        "",
    ]

    if result.dry_run:
        lines.append("_Dry run — nothing was written to Zotero._")
        lines.append("")
    if not added:
        lines.append("_No papers met the score threshold._")
        lines.append("")
    for paper in added:
        lines.extend(
            [
                f"### {paper.title}",
                f"- **Authors:** {_format_authors(paper.authors)}",
                f"- **Journal:** {paper.journal or 'N/A'}",
                f"- **Year:** {paper.year or 'N/A'}",
                f"- **Score:** {paper.score}",
                f"- **Tags:** {', '.join(paper.tags) if paper.tags else 'N/A'}",
                f"- **Reason:** {paper.reason}",
                f"- **Summary:** {paper.summary}",
                f"- **DOI:** {paper.doi or 'N/A'}",
                "",
            ]
        )

    lines.extend(["---", "", "## Ignored Papers", "", "| Title | Score |", "|---|---|"])
    for paper in sorted(result.ignored, key=lambda p: p.score, reverse=True):
        safe_title = paper.title.replace("|", "\\|")
        lines.append(f"| {safe_title} | {paper.score} |")
    lines.append("")

    return "\n".join(lines)


def _render_json(result: PipelineResult) -> str:
    """Render the JSON report body.

    Args:
        result: The pipeline result.

    Returns:
        A JSON string with statistics and the full ranked paper list.
    """
    def paper_dict(paper: RankedPaper) -> dict:
        d = dataclasses.asdict(paper)
        d.pop("raw", None)
        d["added"] = paper.title in set(result.added_titles)
        return d

    payload = {
        "date": result.date,
        "statistics": {
            "fetched": result.fetched,
            "duplicates_removed": result.duplicates_removed,
            "evaluated": result.evaluated,
            "added": result.added_count,
            "ignored": len(result.ignored),
            "average_score": round(result.average_score, 2),
        },
        "dry_run": result.dry_run,
        "papers": [paper_dict(p) for p in result.ranked],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def generate_report(result: PipelineResult, config: PipelineConfig) -> PipelineResult:
    """Write Markdown and JSON reports for a run.

    Args:
        result: The pipeline result to report on.
        config: The pipeline configuration.

    Returns:
        The same result, updated with report paths.
    """
    reports_dir = config.project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    md_path = reports_dir / f"{result.date}.md"
    json_path = reports_dir / f"{result.date}.json"

    md_path.write_text(_render_markdown(result), encoding="utf-8")
    json_path.write_text(_render_json(result), encoding="utf-8")

    result.markdown_path = str(md_path)
    result.json_path = str(json_path)
    logger.info("Wrote report: %s", md_path)
    return result


def latest_report_path(config: PipelineConfig) -> Path | None:
    """Return the path to the most recent JSON report, if any.

    Args:
        config: The pipeline configuration.

    Returns:
        Path to the newest ``reports/*.json`` file, or ``None``.
    """
    reports_dir = config.project_root / "reports"
    candidates = sorted(reports_dir.glob("*.json"))
    return candidates[-1] if candidates else None


def load_result_from_json(path: Path) -> PipelineResult:
    """Reconstruct a :class:`PipelineResult` from a JSON report.

    Used by ``--report-only`` to regenerate a Markdown report without re-running
    the pipeline.

    Args:
        path: Path to a JSON report.

    Returns:
        The reconstructed :class:`PipelineResult`.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data.get("statistics", {})
    added_titles = [p["title"] for p in data.get("papers", []) if p.get("added")]
    ranked = [
        RankedPaper(
            title=p.get("title", ""),
            abstract=p.get("abstract"),
            doi=p.get("doi"),
            authors=p.get("authors", []),
            year=p.get("year"),
            journal=p.get("journal"),
            openalex_id=p.get("openalex_id", ""),
            citation_count=p.get("citation_count", 0),
            url=p.get("url", ""),
            source=p.get("source", ""),
            score=p.get("score", 0),
            reason=p.get("reason", ""),
            tags=p.get("tags", []),
            summary=p.get("summary", ""),
        )
        for p in data.get("papers", [])
    ]
    min_score = min(
        (p.score for p in ranked if p.title in set(added_titles)), default=4
    )
    return PipelineResult(
        date=data.get("date", dt.date.today().isoformat()),
        fetched=stats.get("fetched", 0),
        duplicates_removed=stats.get("duplicates_removed", 0),
        evaluated=stats.get("evaluated", 0),
        ranked=ranked,
        added_titles=added_titles,
        min_score=min_score,
        dry_run=data.get("dry_run", False),
    )
