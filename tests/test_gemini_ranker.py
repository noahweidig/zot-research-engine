"""Unit tests for pipeline.gemini_ranker. All Gemini API calls are mocked."""

from __future__ import annotations

import time
import types

import pytest
from conftest import make_paper
from pydantic import ValidationError

import pipeline.gemini_ranker as gr
from config import (
    GeminiConfig,
    PipelineConfig,
    PipelineSettings,
    PrefilterConfig,
    RelevanceConfig,
    SearchConfig,
    Secrets,
    SourcesConfig,
    ZoteroConfig,
)
from pathlib import Path
from pipeline.gemini_ranker import GeminiRanking, rank_papers


def _config(
    min_score: int = 4,
    concurrency: int = 2,
    fallback: bool = False,
) -> PipelineConfig:
    """Build a minimal in-memory PipelineConfig for tests."""
    return PipelineConfig(
        search=SearchConfig(queries=["x"], days_back=7, max_results_per_query=10),
        relevance=RelevanceConfig(min_score=min_score, interests="GIS"),
        zotero=ZoteroConfig(library_type="user", inbox_collection="AI Inbox"),
        sources=SourcesConfig(
            enable_semantic_scholar=False,
            enable_arxiv=False,
            enable_crossref=False,
        ),
        pipeline=PipelineSettings(
            log_to_file=False, log_file="x.log", gemini_max_concurrency=concurrency
        ),
        prefilter=PrefilterConfig(enabled=True, top_k=25, min_similarity=0.0),
        gemini=GeminiConfig(
            model="gemini-2.0-flash",
            temperature=0.0,
            requests_per_minute=0,
            fallback_to_prefilter=fallback,
        ),
        secrets=Secrets(
            gemini_api_key="k",
            zotero_api_key="k",
            zotero_user_id="1",
            zotero_library_type="user",
            openalex_email="a@b.c",
        ),
        project_root=Path("."),
    )


class _FakeModels:
    """Fake ``client.models`` implementing generate_content."""

    def __init__(self, ranking: GeminiRanking, fail_times: int = 0) -> None:
        self._ranking = ranking
        self._fail_times = fail_times
        self.calls = 0

    def generate_content(self, **_kwargs):  # noqa: ANN003
        self.calls += 1
        if self.calls <= self._fail_times:
            raise Exception("429 RESOURCE_EXHAUSTED: rate limit")
        return types.SimpleNamespace(parsed=self._ranking, text="{}")


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch) -> None:
    """Neutralize tenacity backoff sleeps so tests run instantly."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def test_score_out_of_range_is_rejected() -> None:
    """The response schema rejects scores outside 1-5."""
    with pytest.raises(ValidationError):
        GeminiRanking(score=7, reason="r", tags=[], summary="s")
    with pytest.raises(ValidationError):
        GeminiRanking(score=0, reason="r", tags=[], summary="s")


def test_valid_score_parses() -> None:
    """A well-formed ranking validates."""
    r = GeminiRanking(score=5, reason="r", tags=["gis"], summary="s")
    assert r.score == 5


def test_rank_and_threshold_filter(monkeypatch) -> None:
    """Only papers scoring >= min_score pass the downstream threshold filter."""
    ranking = GeminiRanking(score=5, reason="r", tags=["gis"], summary="s")
    fake = _FakeModels(ranking)
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake))

    ranked = rank_papers([make_paper(title="P1")], _config(min_score=4))
    assert len(ranked) == 1
    assert ranked[0].score == 5

    passing = [p for p in ranked if p.score >= 4]
    assert len(passing) == 1

    # A score below the threshold would be filtered out downstream.
    low = GeminiRanking(score=2, reason="r", tags=[], summary="s")
    fake_low = _FakeModels(low)
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake_low))
    ranked_low = rank_papers([make_paper(title="P2")], _config(min_score=4))
    assert [p for p in ranked_low if p.score >= 4] == []


def test_papers_without_abstract_never_reach_ranker(monkeypatch) -> None:
    """Papers lacking an abstract are skipped before any API call."""
    ranking = GeminiRanking(score=5, reason="r", tags=[], summary="s")
    fake = _FakeModels(ranking)
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake))

    ranked = rank_papers([make_paper(title="NoAbs", abstract=None)], _config())
    assert ranked == []
    assert fake.calls == 0


def test_retry_on_429(monkeypatch) -> None:
    """A 429 error is retried and the call eventually succeeds."""
    ranking = GeminiRanking(score=4, reason="r", tags=[], summary="s")
    fake = _FakeModels(ranking, fail_times=2)
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake))

    ranked = rank_papers([make_paper(title="Retry")], _config())
    assert len(ranked) == 1
    assert fake.calls == 3  # 2 failures + 1 success


def test_failure_without_fallback_drops_paper(monkeypatch) -> None:
    """With fallback disabled, an exhausted-quota paper is dropped (None)."""
    ranking = GeminiRanking(score=5, reason="r", tags=[], summary="s")
    # Always fail more than GEMINI_MAX_ATTEMPTS so retries are exhausted.
    fake = _FakeModels(ranking, fail_times=99)
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake))

    ranked = rank_papers([make_paper(title="Dead")], _config(fallback=False))
    assert ranked == []


def test_fallback_to_prefilter_when_gemini_unavailable(monkeypatch) -> None:
    """With fallback enabled, a failed paper is scored from the prefilter."""
    ranking = GeminiRanking(score=5, reason="r", tags=[], summary="s")
    fake = _FakeModels(ranking, fail_times=99)  # Gemini never succeeds.
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake))

    paper = make_paper(title="Fallback", abstract="wildfire risk in the eastern US")
    scores = {id(paper): 0.7}  # High relevance -> heuristic score 4.
    ranked = rank_papers([paper], _config(fallback=True), prefilter_scores=scores)

    assert len(ranked) == 1
    assert ranked[0].score == 4
    assert gr.HEURISTIC_TAG in ranked[0].tags


def test_heuristic_score_mapping() -> None:
    """Normalized similarity maps to a conservative 1-4 heuristic score."""
    assert gr._heuristic_score(0.9) == 4
    assert gr._heuristic_score(0.5) == 3
    assert gr._heuristic_score(0.2) == 2
    assert gr._heuristic_score(0.0) == 1


def test_rate_limiter_spaces_calls(monkeypatch) -> None:
    """The rate limiter enforces a minimum interval between call starts."""
    sleeps: list[float] = []
    monkeypatch.setattr(gr.time, "sleep", lambda s: sleeps.append(s))

    limiter = gr._RateLimiter(requests_per_minute=60)  # 1s interval.
    limiter.acquire()  # First call is immediate.
    limiter.acquire()  # Second must wait ~1s.
    assert any(s > 0 for s in sleeps)
