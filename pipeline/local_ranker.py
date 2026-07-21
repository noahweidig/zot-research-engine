"""Free, offline relevance ranking.

Scores every candidate paper against the researcher's interests using a
dependency-free lexical model — no API calls, no quota, no cost. This is the
pipeline's primary ranker: because it runs entirely locally it can never be
zeroed out by an exhausted Gemini free-tier quota, which is what used to leave
runs with zero evaluated papers.

The score is *IDF-weighted interest coverage*: the fraction of the researcher's
distinctive interest terms (weighted so that terms which are rare across the
day's candidate set count for more) that a paper's title and abstract cover.
Title matches are boosted because a term in the title is a stronger relevance
signal than one buried in the abstract. Coverage in ``[0, 1]`` is then bucketed
onto the familiar 1-5 scale via configurable thresholds.

Gemini is still used afterwards, but only to enrich the small shortlist that
clears the score threshold (see :mod:`pipeline.gemini_ranker`).
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from config import PipelineConfig
from pipeline.models import Paper, RankedPaper

logger = logging.getLogger(__name__)

# Title matches weigh this much more than abstract matches.
_TITLE_BOOST = 2.0
# Number of matched terms to surface as tags / in the reason string.
_MAX_TAGS = 5

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small, domain-agnostic stopword list. Kept intentionally short: IDF
# weighting already suppresses terms that are common across the candidate set,
# so this only needs to drop function words that would otherwise form noisy
# bigrams (e.g. "of large", "in the").
_STOPWORDS = frozenset(
    """
    a an the and or of for to in on at by with from into over under as is are was
    were be been being this that these those it its their our your his her they we
    you i he she them us can could would should may might will shall do does did
    not no nor so than then there here which who whom whose what when where why how
    """.split()
)


def _tokens(text: str) -> list[str]:
    """Tokenize text into lowercased, de-stopworded content tokens.

    Args:
        text: Arbitrary input text.

    Returns:
        Content tokens (length >= 3, not stopwords), in order.
    """
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    ]


def _grams(tokens: list[str]) -> list[str]:
    """Expand a token list into unigrams plus adjacent bigrams.

    Bigrams let the scorer reward multi-word interest phrases (``"prescribed
    burning"``) far more than the two words appearing separately.

    Args:
        tokens: Content tokens.

    Returns:
        Unigrams followed by bigrams.
    """
    grams = list(tokens)
    grams.extend(f"{a} {b}" for a, b in zip(tokens, tokens[1:]))
    return grams


def _bucket_score(coverage: float, thresholds: list[float]) -> int:
    """Map a coverage fraction in ``[0, 1]`` to a 1-5 score.

    Args:
        coverage: IDF-weighted interest coverage.
        thresholds: Four descending cutoffs ``[t5, t4, t3, t2]``; coverage at or
            above ``t5`` scores 5, above ``t4`` scores 4, and so on. Below the
            last cutoff scores 1.

    Returns:
        An integer score from 1 to 5.
    """
    t5, t4, t3, t2 = thresholds
    if coverage >= t5:
        return 5
    if coverage >= t4:
        return 4
    if coverage >= t3:
        return 3
    if coverage >= t2:
        return 2
    return 1


def _summarize(abstract: str, max_chars: int = 320) -> str:
    """Produce a short plain-text summary from an abstract.

    Uses the first sentences of the abstract, trimmed to ``max_chars``. This is
    the free fallback summary; Gemini may replace it for shortlisted papers.

    Args:
        abstract: The paper abstract.
        max_chars: Approximate maximum summary length.

    Returns:
        A trimmed summary string.
    """
    text = " ".join(abstract.split())
    if len(text) <= max_chars:
        return text
    # Prefer to cut at a sentence boundary before the limit.
    window = text[: max_chars + 1]
    cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if cut >= max_chars // 2:
        return window[: cut + 1].strip()
    return text[:max_chars].rstrip() + "…"


def rank_papers(papers: list[Paper], config: PipelineConfig) -> list[RankedPaper]:
    """Score all candidate papers locally, with no external API calls.

    Papers without an abstract are skipped defensively (they should already have
    been filtered upstream). IDF is computed over the day's candidate set, so
    "relevance" is judged relative to what is actually on offer that day.

    Args:
        papers: The candidate papers to score.
        config: The pipeline configuration.

    Returns:
        Ranked papers, sorted by descending score (ties broken by coverage).
    """
    candidates = [p for p in papers if p.abstract]
    skipped = len(papers) - len(candidates)
    if skipped:
        logger.warning("Skipping %d paper(s) with no abstract before ranking", skipped)
    if not candidates:
        return []

    # Per-paper gram sets (title kept separate so title hits can be boosted).
    title_grams: list[set[str]] = []
    body_grams: list[set[str]] = []
    for paper in candidates:
        tg = set(_grams(_tokens(paper.title)))
        bg = set(_grams(_tokens(paper.abstract or "")))
        title_grams.append(tg)
        body_grams.append(bg)

    # Document frequency across the candidate set, then smoothed IDF.
    n_docs = len(candidates)
    doc_freq: Counter[str] = Counter()
    for tg, bg in zip(title_grams, body_grams):
        for gram in tg | bg:
            doc_freq[gram] += 1
    idf = {
        gram: math.log((n_docs + 1) / (df + 1)) + 1.0 for gram, df in doc_freq.items()
    }

    # Interest terms that actually occur in the corpus define the achievable
    # weight; terms absent from every paper cannot discriminate and are dropped.
    interest_grams = [g for g in _grams(_tokens(config.relevance.interests)) if g in idf]
    total_weight = sum(idf[g] for g in set(interest_grams))
    thresholds = config.ranker.thresholds

    ranked: list[tuple[float, RankedPaper]] = []
    for paper, tg, bg in zip(candidates, title_grams, body_grams):
        matched_weight = 0.0
        matched: list[str] = []
        for gram in set(interest_grams):
            if gram in tg:
                matched_weight += idf[gram] * _TITLE_BOOST
                matched.append(gram)
            elif gram in bg:
                matched_weight += idf[gram]
                matched.append(gram)

        coverage = 0.0 if total_weight == 0 else min(matched_weight / total_weight, 1.0)
        score = _bucket_score(coverage, thresholds)

        top = sorted(matched, key=lambda g: idf[g], reverse=True)[:_MAX_TAGS]
        tags = [g.title() for g in top]
        n_interest = len(set(interest_grams))
        reason = (
            f"Lexical relevance {coverage:.0%}: matched "
            f"{len(matched)}/{n_interest} key interest terms"
        )
        if top:
            reason += " (" + ", ".join(top) + ")"
        summary = _summarize(paper.abstract or "")

        ranked.append(
            (
                coverage,
                RankedPaper.from_paper(
                    paper,
                    score=score,
                    reason=reason,
                    tags=tags,
                    summary=summary,
                ),
            )
        )

    ranked.sort(key=lambda item: (item[1].score, item[0]), reverse=True)
    result = [rp for _, rp in ranked]
    logger.info("Locally ranked %d paper(s)", len(result))
    return result
