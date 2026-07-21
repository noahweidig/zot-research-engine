"""Gemini relevance ranking.

Scores each candidate paper against the researcher's interests using the
``google-genai`` SDK (google.genai v1.x). Output is enforced at the API level
via a typed ``response_schema`` — the model returns structured JSON that maps
directly onto :class:`GeminiRanking`, so no free-text JSON parsing is needed.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from config import PipelineConfig
from pipeline.models import Paper, RankedPaper

logger = logging.getLogger(__name__)

GEMINI_MAX_ATTEMPTS = 5

# Tag applied to papers scored heuristically (free prefilter) when Gemini is
# unavailable, so they are clearly distinguishable in the report and in Zotero.
HEURISTIC_TAG = "Heuristic Score"


class _RateLimiter:
    """Thread-safe minimum-interval limiter to respect Gemini's free RPM cap.

    The free Gemini tier allows only a handful of requests per minute. Spacing
    call *starts* by a fixed minimum interval keeps concurrent workers from
    bursting past that cap (the leading cause of ``429`` quota errors).
    """

    def __init__(self, requests_per_minute: int) -> None:
        """Initialize the limiter.

        Args:
            requests_per_minute: Maximum request starts per minute. Values <= 0
                disable throttling.
        """
        self._min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        """Block until the caller is allowed to start its next request."""
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            start = max(now, self._next_allowed)
            self._next_allowed = start + self._min_interval
        if wait > 0:
            time.sleep(wait)


def _heuristic_score(similarity: float) -> int:
    """Map a normalized prefilter similarity in ``[0, 1]`` to a 1-5 score.

    Deliberately conservative: a heuristic (non-Gemini) match never claims the
    top "must read" score of 5.

    Args:
        similarity: Normalized BM25 relevance score from the prefilter.

    Returns:
        An integer score from 1 to 4.
    """
    if similarity >= 0.6:
        return 4
    if similarity >= 0.35:
        return 3
    if similarity >= 0.15:
        return 2
    return 1

SYSTEM_PROMPT = """\
You are a research assistant helping a scientist curate academic literature.
Score the relevance of the paper below to the researcher's interests.
Return only valid JSON matching the schema. Do not add commentary.

Researcher interests: {interests}

Scoring guide:
5 = Must read: directly addresses core research interests
4 = Very relevant: closely related, high value
3 = Interesting: tangentially related
2 = Low relevance: minor overlap
1 = Ignore: not relevant

Paper title: {title}
Abstract: {abstract}
"""


class GeminiRanking(BaseModel):
    """Structured ranking returned by Gemini for a single paper."""

    score: int = Field(ge=1, le=5, description="Relevance score from 1 to 5")
    reason: str = Field(description="Why this score was assigned")
    tags: list[str] = Field(description="Relevant topic tags")
    summary: str = Field(description="2-3 sentence plain-language summary")


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Return whether an exception represents a rate-limit (429) error.

    Args:
        exc: The raised exception.

    Returns:
        ``True`` if the error is a retryable 429 rate-limit error.
    """
    if isinstance(exc, genai_errors.APIError):
        return getattr(exc, "code", None) == 429
    return "429" in str(exc)


def _build_prompt(paper: Paper, interests: str) -> str:
    """Render the ranking prompt for a paper.

    Args:
        paper: The paper to score.
        interests: The researcher's interests string.

    Returns:
        The fully rendered prompt.
    """
    return SYSTEM_PROMPT.format(
        interests=interests,
        title=paper.title,
        abstract=paper.abstract or "",
    )


def _heuristic_ranked(paper: Paper, similarity: float) -> RankedPaper:
    """Build a heuristic :class:`RankedPaper` from the free prefilter score.

    Used when Gemini cannot score a paper (e.g. quota exhausted) so the run
    still yields a useful, relevance-ranked report instead of nothing.

    Args:
        paper: The paper to score.
        similarity: Normalized BM25 relevance score in ``[0, 1]``.

    Returns:
        A :class:`RankedPaper` scored from the local prefilter, tagged so it is
        clearly distinguishable from a genuine Gemini score.
    """
    return RankedPaper.from_paper(
        paper,
        score=_heuristic_score(similarity),
        reason=(
            "Scored by the free local relevance filter (BM25) because Gemini was "
            f"unavailable. Lexical relevance to interests: {similarity:.2f}."
        ),
        tags=[HEURISTIC_TAG],
        summary=paper.abstract[:280].strip() if paper.abstract else "",
    )


def _rank_one(
    client: genai.Client,
    paper: Paper,
    config: PipelineConfig,
    semaphore: threading.Semaphore,
    limiter: _RateLimiter,
    similarity: float,
) -> RankedPaper | None:
    """Score a single paper, retrying on rate-limit errors.

    Args:
        client: The Gemini client.
        paper: The paper to score.
        config: The pipeline configuration.
        semaphore: Concurrency limiter for Gemini calls.
        limiter: Minimum-interval limiter enforcing the free-tier RPM cap.
        similarity: Normalized prefilter relevance score, used for the heuristic
            fallback if Gemini scoring fails.

    Returns:
        A :class:`RankedPaper` (from Gemini, or from the heuristic fallback when
        enabled), or ``None`` if scoring failed and fallback is disabled.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(GEMINI_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=2, max=60),
        retry=retry_if_exception(_is_rate_limit_error),
        before_sleep=lambda state: logger.warning(
            "Gemini rate limited, retrying (attempt %d): %s",
            state.attempt_number,
            paper.title,
        ),
    )
    def _call() -> GeminiRanking:
        limiter.acquire()
        response = client.models.generate_content(
            model=config.gemini.model,
            contents=_build_prompt(paper, config.relevance.interests),
            config=genai_types.GenerateContentConfig(
                temperature=config.gemini.temperature,
                response_mime_type="application/json",
                response_schema=GeminiRanking,
            ),
        )
        parsed = response.parsed
        if isinstance(parsed, GeminiRanking):
            return parsed
        # Fall back to validating the raw JSON text if the SDK did not parse it.
        return GeminiRanking.model_validate_json(response.text)

    with semaphore:
        try:
            ranking = _call()
        except Exception as exc:  # noqa: BLE001 - one failure must not abort the run
            if config.gemini.fallback_to_prefilter:
                logger.warning(
                    "Gemini scoring failed for %r (%s); using free prefilter score.",
                    paper.title,
                    exc,
                )
                return _heuristic_ranked(paper, similarity)
            logger.exception("Failed to rank paper %r", paper.title)
            return None

    logger.info("Scored %d: %s", ranking.score, paper.title)
    return RankedPaper.from_paper(
        paper,
        score=ranking.score,
        reason=ranking.reason,
        tags=ranking.tags,
        summary=ranking.summary,
    )


def rank_papers(
    papers: list[Paper],
    config: PipelineConfig,
    prefilter_scores: dict[int, float] | None = None,
) -> list[RankedPaper]:
    """Score all candidate papers with Gemini.

    Papers without an abstract are skipped defensively (they should already have
    been filtered out upstream). Calls are made concurrently, bounded by a
    semaphore and spaced by a rate limiter, to avoid exhausting the free-tier
    quota. If Gemini is unavailable and ``gemini.fallback_to_prefilter`` is set,
    papers are scored from the free local prefilter instead of being dropped.

    Args:
        papers: The candidate papers to score.
        config: The pipeline configuration.
        prefilter_scores: Optional mapping of ``id(paper)`` to a normalized
            ``[0, 1]`` relevance score, used for the heuristic fallback.

    Returns:
        Ranked papers, sorted by descending score.
    """
    scores = prefilter_scores or {}
    candidates = [p for p in papers if p.abstract]
    skipped = len(papers) - len(candidates)
    if skipped:
        logger.warning("Skipping %d paper(s) with no abstract before ranking", skipped)
    if not candidates:
        return []

    client = genai.Client(api_key=config.secrets.gemini_api_key)
    semaphore = threading.Semaphore(config.pipeline.gemini_max_concurrency)
    limiter = _RateLimiter(config.gemini.requests_per_minute)

    ranked: list[RankedPaper] = []
    with ThreadPoolExecutor(
        max_workers=config.pipeline.gemini_max_concurrency
    ) as executor:
        futures = [
            executor.submit(
                _rank_one,
                client,
                paper,
                config,
                semaphore,
                limiter,
                scores.get(id(paper), 0.0),
            )
            for paper in candidates
        ]
        for future in futures:
            result = future.result()
            if result is not None:
                ranked.append(result)

    ranked.sort(key=lambda p: p.score, reverse=True)
    logger.info("Ranked %d/%d paper(s) successfully", len(ranked), len(candidates))
    return ranked
