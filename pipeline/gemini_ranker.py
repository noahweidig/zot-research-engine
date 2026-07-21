"""Gemini enrichment for shortlisted papers.

The heavy lifting of relevance scoring is done for free and offline by
:mod:`pipeline.local_ranker`. Gemini is reserved for the "small things" it does
best without burning through the free-tier quota: polishing the handful of
papers that clear the score threshold with a fluent plain-language summary, a
one-line relevance rationale, and topic tags.

Because only the shortlist is sent — typically a few papers per day rather than
every candidate — a normal daily run stays comfortably inside the free tier.
Enrichment is best-effort: any failure (quota, network, bad response) leaves the
paper's local summary/reason/tags untouched, so the pipeline never fails here.
"""

from __future__ import annotations

import logging
import threading
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

GEMINI_MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """\
You are a research assistant helping a scientist triage academic literature.
The paper below has already been judged relevant. Write a concise, accurate
enrichment for it. Return only valid JSON matching the schema; no commentary.

Researcher interests: {interests}

Paper title: {title}
Abstract: {abstract}
"""


class GeminiEnrichment(BaseModel):
    """Structured enrichment returned by Gemini for a single shortlisted paper."""

    reason: str = Field(description="One sentence on why this paper is relevant")
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
    """Render the enrichment prompt for a paper.

    Args:
        paper: The paper to enrich.
        interests: The researcher's interests string.

    Returns:
        The fully rendered prompt.
    """
    return SYSTEM_PROMPT.format(
        interests=interests,
        title=paper.title,
        abstract=paper.abstract or "",
    )


def _enrich_one(
    client: genai.Client,
    paper: RankedPaper,
    config: PipelineConfig,
    semaphore: threading.Semaphore,
) -> None:
    """Enrich a single shortlisted paper in place, best-effort.

    On any failure the paper keeps its local summary/reason/tags, so callers can
    ignore the outcome — nothing is lost when Gemini is unavailable.

    Args:
        client: The Gemini client.
        paper: The shortlisted paper to enrich (mutated in place on success).
        config: The pipeline configuration.
        semaphore: Concurrency limiter for Gemini calls.
    """

    @retry(
        reraise=True,
        stop=stop_after_attempt(GEMINI_MAX_ATTEMPTS),
        wait=wait_exponential_jitter(initial=2, max=30),
        retry=retry_if_exception(_is_rate_limit_error),
        before_sleep=lambda state: logger.warning(
            "Gemini rate limited, retrying (attempt %d): %s",
            state.attempt_number,
            paper.title,
        ),
    )
    def _call() -> GeminiEnrichment:
        response = client.models.generate_content(
            model=config.gemini.model,
            contents=_build_prompt(paper, config.relevance.interests),
            config=genai_types.GenerateContentConfig(
                temperature=config.gemini.temperature,
                response_mime_type="application/json",
                response_schema=GeminiEnrichment,
            ),
        )
        parsed = response.parsed
        if isinstance(parsed, GeminiEnrichment):
            return parsed
        return GeminiEnrichment.model_validate_json(response.text)

    with semaphore:
        try:
            enrichment = _call()
        except Exception as exc:  # noqa: BLE001 - enrichment must never abort the run
            logger.warning(
                "Gemini enrichment failed for %r; keeping local summary: %s",
                paper.title,
                exc,
            )
            return

    paper.reason = enrichment.reason
    if enrichment.tags:
        paper.tags = enrichment.tags
    if enrichment.summary:
        paper.summary = enrichment.summary
    logger.info("Enriched: %s", paper.title)


def enrich_papers(
    papers: list[RankedPaper], config: PipelineConfig
) -> list[RankedPaper]:
    """Enrich a shortlist of already-ranked papers with Gemini, best-effort.

    Only the shortlist should be passed (papers that cleared the score
    threshold), keeping the number of Gemini calls small enough for the free
    tier. Papers are mutated in place; the same list is returned for convenience.

    Args:
        papers: The shortlisted ranked papers to enrich.
        config: The pipeline configuration.

    Returns:
        The same list, with summaries/reasons/tags upgraded where Gemini
        succeeded.
    """
    if not papers:
        return papers

    try:
        client = genai.Client(api_key=config.secrets.gemini_api_key)
    except Exception as exc:  # noqa: BLE001 - fall back to local enrichment
        logger.warning("Could not init Gemini client; skipping enrichment: %s", exc)
        return papers

    semaphore = threading.Semaphore(config.pipeline.gemini_max_concurrency)
    with ThreadPoolExecutor(
        max_workers=config.pipeline.gemini_max_concurrency
    ) as executor:
        futures = [
            executor.submit(_enrich_one, client, paper, config, semaphore)
            for paper in papers
        ]
        for future in futures:
            future.result()

    logger.info("Gemini enrichment complete for %d paper(s)", len(papers))
    return papers
