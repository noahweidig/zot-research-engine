"""Free, local relevance pre-filtering (no API key, no network, no quota).

Gemini's free tier has tight per-minute and per-day quotas. Sending *every*
deduplicated candidate to Gemini quickly exhausts that quota, at which point
every request fails with ``429 RESOURCE_EXHAUSTED`` and the whole run produces
nothing.

This module fixes that by ranking candidates *before* they ever reach Gemini,
using a classic BM25 lexical relevance score of each paper (title + abstract)
against the researcher's interests. BM25 is the same free relevance-ranking
technique that open scholarly tools such as Semantic Scholar and CORE expose in
their search APIs — but computed locally here, so it costs nothing, needs no
API key, and can never rate-limit. Only the most promising papers are then
forwarded to Gemini for the expensive fine-grained 1-5 scoring.

The score is normalized to ``[0, 1]`` (relative to the best-matching paper in
the batch) so it can also serve as a heuristic fallback when Gemini is
unavailable — see :mod:`pipeline.gemini_ranker`.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter

from config import PipelineConfig
from pipeline.models import Paper

logger = logging.getLogger(__name__)

# BM25 free parameters (standard, well-tested defaults).
_BM25_K1 = 1.5
_BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A small English stopword list — enough to stop common words from dominating
# the lexical match without pulling in a heavyweight NLP dependency.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be by for from has have in into is it its of on or that
    the their there these this to was were will with we our can may using use
    used based study paper results show shows present presents approach method
    methods between during over under about also than then them they which while
    """.split()
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split into alphanumeric tokens, and drop stopwords.

    Args:
        text: Arbitrary input text.

    Returns:
        The list of content tokens (length >= 2, non-stopword).
    """
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) > 1 and tok not in _STOPWORDS
    ]


def _paper_text(paper: Paper) -> str:
    """Return the text used to represent a paper for lexical matching."""
    return f"{paper.title or ''} {paper.abstract or ''}"


def _build_query_terms(config: PipelineConfig) -> list[str]:
    """Build the relevance query from interests plus search queries.

    The search queries are included because they encode what the researcher
    deliberately went looking for, reinforcing the free-text interests.

    Args:
        config: The pipeline configuration.

    Returns:
        The deduplicated list of query tokens.
    """
    parts = [config.relevance.interests, *config.search.queries]
    return _tokenize(" ".join(parts))


def score_relevance(papers: list[Paper], config: PipelineConfig) -> dict[int, float]:
    """Compute a normalized BM25 relevance score for each paper.

    Args:
        papers: The candidate papers.
        config: The pipeline configuration (interests + queries drive the query).

    Returns:
        A mapping of ``id(paper)`` to a relevance score in ``[0, 1]``, where 1 is
        the best-matching paper in the batch. Empty if there is nothing to score.
    """
    if not papers:
        return {}

    query_terms = _build_query_terms(config)
    if not query_terms:
        logger.warning("Prefilter: empty interests/queries; keeping all papers.")
        return {id(p): 1.0 for p in papers}

    docs = [_tokenize(_paper_text(p)) for p in papers]
    doc_lengths = [len(d) for d in docs]
    avg_len = sum(doc_lengths) / len(docs) if docs else 0.0

    # Document frequency for each query term.
    doc_freq: Counter[str] = Counter()
    unique_query_terms = set(query_terms)
    for doc in docs:
        present = unique_query_terms.intersection(doc)
        for term in present:
            doc_freq[term] += 1

    n_docs = len(docs)
    idf: dict[str, float] = {}
    for term in unique_query_terms:
        df = doc_freq.get(term, 0)
        # BM25 idf with +1 smoothing so it is always positive.
        idf[term] = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))

    raw_scores: list[float] = []
    for doc, length in zip(docs, doc_lengths):
        counts = Counter(doc)
        score = 0.0
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * (length / avg_len if avg_len else 0))
        for term in unique_query_terms:
            tf = counts.get(term, 0)
            if tf == 0:
                continue
            score += idf[term] * (tf * (_BM25_K1 + 1)) / (tf + norm)
        raw_scores.append(score)

    top = max(raw_scores) if raw_scores else 0.0
    if top <= 0:
        # No lexical overlap at all — do not falsely rank anything; treat as ties.
        return {id(p): 0.0 for p in papers}

    return {id(p): raw / top for p, raw in zip(papers, raw_scores)}


def prefilter_papers(
    papers: list[Paper], config: PipelineConfig
) -> tuple[list[Paper], dict[int, float]]:
    """Rank candidates locally and keep only the most relevant ones.

    Reduces the number of papers sent to Gemini to ``prefilter.top_k`` and drops
    anything below ``prefilter.min_similarity``, so Gemini's scarce free-tier
    quota is spent only on genuinely promising papers.

    Args:
        papers: The deduplicated candidate papers.
        config: The pipeline configuration.

    Returns:
        A tuple of ``(kept_papers, scores)`` where ``scores`` maps ``id(paper)``
        to the normalized relevance score for *every* input paper (used later as
        a Gemini fallback), and ``kept_papers`` is the filtered, relevance-sorted
        subset. If prefiltering is disabled, all papers are returned unchanged
        with uniform scores.
    """
    scores = score_relevance(papers, config)

    if not config.prefilter.enabled:
        return papers, scores

    ranked = sorted(papers, key=lambda p: scores.get(id(p), 0.0), reverse=True)

    min_sim = config.prefilter.min_similarity
    kept = [p for p in ranked if scores.get(id(p), 0.0) >= min_sim]
    kept = kept[: config.prefilter.top_k]

    dropped = len(papers) - len(kept)
    if dropped > 0:
        logger.info(
            "Prefilter: kept %d/%d paper(s) for Gemini (dropped %d below "
            "top_k=%d / min_similarity=%.2f)",
            len(kept),
            len(papers),
            dropped,
            config.prefilter.top_k,
            min_sim,
        )
    else:
        logger.info("Prefilter: all %d paper(s) within budget for Gemini", len(kept))
    return kept, scores
