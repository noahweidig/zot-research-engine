"""Unit tests for pipeline.prefilter (free local BM25 relevance ranking)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_paper

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
from pipeline.prefilter import prefilter_papers, score_relevance


def _config(
    *,
    interests: str = "wildfire eastern united states prescribed burning",
    queries: list[str] | None = None,
    enabled: bool = True,
    top_k: int = 25,
    min_similarity: float = 0.0,
) -> PipelineConfig:
    """Build a minimal PipelineConfig for prefilter tests."""
    return PipelineConfig(
        search=SearchConfig(
            queries=queries if queries is not None else ["large wildfire"],
            days_back=7,
            max_results_per_query=10,
        ),
        relevance=RelevanceConfig(min_score=4, interests=interests),
        zotero=ZoteroConfig(library_type="user", inbox_collection="AI Inbox"),
        sources=SourcesConfig(
            enable_semantic_scholar=False,
            enable_arxiv=False,
            enable_crossref=False,
        ),
        pipeline=PipelineSettings(
            log_to_file=False, log_file="x.log", gemini_max_concurrency=2
        ),
        prefilter=PrefilterConfig(
            enabled=enabled, top_k=top_k, min_similarity=min_similarity
        ),
        gemini=GeminiConfig(
            model="gemini-2.0-flash",
            temperature=0.0,
            requests_per_minute=0,
            fallback_to_prefilter=True,
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


def test_relevant_paper_scores_higher_than_irrelevant() -> None:
    """A paper matching the interests outranks an unrelated one."""
    relevant = make_paper(
        title="Drivers of large wildfires in the eastern United States",
        abstract="We analyze prescribed burning and wildfire risk in eastern US forests.",
        openalex_id="W_rel",
    )
    irrelevant = make_paper(
        title="Quantum entanglement in superconducting qubits",
        abstract="A study of coherence times in transmon quantum processors.",
        openalex_id="W_irrel",
    )
    scores = score_relevance([relevant, irrelevant], _config())
    assert scores[id(relevant)] > scores[id(irrelevant)]
    # Best match is normalized to 1.0 (approx to avoid float-equality pitfalls).
    assert scores[id(relevant)] == pytest.approx(1.0)


def test_top_k_limits_papers_sent_to_gemini() -> None:
    """Only the top_k most relevant papers survive prefiltering."""
    papers = [
        make_paper(
            title=f"wildfire study {i}",
            abstract="prescribed burning wildfire risk eastern united states",
            openalex_id=f"W_rel_{i}",
        )
        for i in range(5)
    ] + [
        make_paper(
            title=f"unrelated topic {i}",
            abstract="marine biology of deep sea coral reefs",
            openalex_id=f"W_off_{i}",
        )
        for i in range(5)
    ]
    kept, scores = prefilter_papers(papers, _config(top_k=3))
    assert len(kept) == 3
    assert len(scores) == 10  # Scores computed for every input paper.
    # The survivors should be the wildfire papers, not the marine ones.
    assert all("wildfire" in p.title for p in kept)


def test_min_similarity_drops_weak_matches() -> None:
    """Papers below min_similarity are dropped even within top_k."""
    strong = make_paper(
        title="wildfire prescribed burning eastern united states",
        abstract="wildfire risk drivers eastern united states prescribed burning",
        openalex_id="W_strong",
    )
    weak = make_paper(
        title="unrelated marine biology",
        abstract="deep sea coral reef ecosystems",
        openalex_id="W_weak",
    )
    kept, _ = prefilter_papers([strong, weak], _config(min_similarity=0.5))
    assert kept == [strong]


def test_disabled_prefilter_keeps_all_papers() -> None:
    """When disabled, prefilter returns every paper untouched."""
    papers = [make_paper(title=f"p{i}", openalex_id=f"W{i}") for i in range(4)]
    kept, scores = prefilter_papers(papers, _config(enabled=False))
    assert len(kept) == 4
    assert len(scores) == 4


def test_empty_input_is_safe() -> None:
    """No papers yields empty results without error."""
    kept, scores = prefilter_papers([], _config())
    assert kept == []
    assert scores == {}
