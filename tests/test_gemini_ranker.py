"""Unit tests for pipeline.gemini_ranker enrichment. Gemini calls are mocked."""

from __future__ import annotations

import time
import types
from pathlib import Path

import pytest
from conftest import make_config, make_ranked

import pipeline.gemini_ranker as gr
from pipeline.gemini_ranker import GeminiEnrichment, enrich_papers


class _FakeModels:
    """Fake ``client.models`` implementing generate_content."""

    def __init__(self, enrichment: GeminiEnrichment, fail_times: int = 0) -> None:
        self._enrichment = enrichment
        self._fail_times = fail_times
        self.calls = 0

    def generate_content(self, **_kwargs):  # noqa: ANN003
        self.calls += 1
        if self.calls <= self._fail_times:
            raise Exception("429 RESOURCE_EXHAUSTED: rate limit")
        return types.SimpleNamespace(parsed=self._enrichment, text="{}")


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch) -> None:
    """Neutralize tenacity backoff sleeps so tests run instantly."""
    monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)


def test_enrichment_overwrites_local_fields(monkeypatch) -> None:
    """A successful Gemini call replaces local reason/tags/summary."""
    enrichment = GeminiEnrichment(
        reason="Directly on-topic", tags=["fire"], summary="Great paper."
    )
    fake = _FakeModels(enrichment)
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake))

    paper = make_ranked(reason="local reason", tags=["local"], summary="local summary")
    enrich_papers([paper], make_config())

    assert paper.reason == "Directly on-topic"
    assert paper.tags == ["fire"]
    assert paper.summary == "Great paper."
    assert fake.calls == 1


def test_enrichment_failure_keeps_local_fields(monkeypatch) -> None:
    """When Gemini keeps failing, the paper's local fields survive untouched."""
    enrichment = GeminiEnrichment(reason="x", tags=["x"], summary="x")
    fake = _FakeModels(enrichment, fail_times=gr.GEMINI_MAX_ATTEMPTS + 5)
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake))

    paper = make_ranked(reason="local reason", tags=["local"], summary="local summary")
    result = enrich_papers([paper], make_config())

    assert result[0].reason == "local reason"
    assert result[0].tags == ["local"]
    assert result[0].summary == "local summary"


def test_empty_shortlist_makes_no_calls(monkeypatch) -> None:
    """An empty shortlist never constructs a client or calls the API."""

    def _boom(**_k):  # pragma: no cover - must not be reached
        raise AssertionError("Gemini client should not be created")

    monkeypatch.setattr(gr.genai, "Client", _boom)
    assert enrich_papers([], make_config()) == []


def test_retry_on_429_then_succeeds(monkeypatch) -> None:
    """A transient 429 is retried and enrichment eventually applies."""
    enrichment = GeminiEnrichment(reason="ok", tags=["ok"], summary="ok")
    fake = _FakeModels(enrichment, fail_times=1)
    monkeypatch.setattr(gr.genai, "Client", lambda **_k: _FakeClient(fake))

    paper = make_ranked(reason="local")
    enrich_papers([paper], make_config())

    assert paper.reason == "ok"
    assert fake.calls == 2


def test_bad_client_init_is_non_fatal(monkeypatch) -> None:
    """If the client cannot be built, enrichment is skipped without raising."""

    def _boom(**_k):
        raise RuntimeError("no key")

    monkeypatch.setattr(gr.genai, "Client", _boom)
    paper = make_ranked(reason="local reason")
    result = enrich_papers([paper], make_config())
    assert result[0].reason == "local reason"


# Path import kept for parity with other test modules that build configs.
_ = Path
