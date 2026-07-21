"""Unit tests for the free, offline pipeline.local_ranker."""

from __future__ import annotations

from conftest import make_config, make_paper

from pipeline.local_ranker import rank_papers


def test_ranks_relevant_above_irrelevant() -> None:
    """A paper matching the interests outscores an off-topic one."""
    interests = "prescribed burning wildfire eastern united states"
    relevant = make_paper(
        title="Prescribed burning reduces wildfire risk in the eastern United States",
        abstract=(
            "We study how prescribed burning affects large wildfire occurrence "
            "across the eastern United States, quantifying fuel and risk."
        ),
    )
    irrelevant = make_paper(
        title="A survey of deep learning for image captioning",
        abstract="Neural networks generate captions for photographs of animals.",
    )

    ranked = rank_papers([irrelevant, relevant], make_config(interests=interests))

    assert ranked[0].title == relevant.title
    assert ranked[0].score > ranked[-1].score
    assert ranked[0].score >= 4


def test_papers_without_abstract_are_skipped() -> None:
    """Papers lacking an abstract never reach the scorer."""
    ranked = rank_papers(
        [make_paper(title="NoAbs", abstract=None)], make_config()
    )
    assert ranked == []


def test_scores_are_in_range_and_sorted() -> None:
    """All scores fall in 1-5 and results are sorted descending."""
    papers = [
        make_paper(title=f"Wildfire study {i}", abstract="wildfire risk in forests")
        for i in range(5)
    ]
    ranked = rank_papers(papers, make_config(interests="wildfire risk"))
    assert all(1 <= p.score <= 5 for p in ranked)
    assert [p.score for p in ranked] == sorted(
        (p.score for p in ranked), reverse=True
    )


def test_thresholds_control_scoring() -> None:
    """Permissive thresholds raise the score of a partially matching paper."""
    paper = make_paper(
        title="Wildfire occurrence",
        abstract="A short note on wildfire occurrence patterns.",
    )
    strict = rank_papers(
        [paper],
        make_config(interests="wildfire occurrence", thresholds=[0.9, 0.8, 0.7, 0.6]),
    )[0]
    lax = rank_papers(
        [paper],
        make_config(interests="wildfire occurrence", thresholds=[0.3, 0.2, 0.1, 0.05]),
    )[0]
    assert lax.score >= strict.score


def test_local_ranker_populates_reason_and_tags() -> None:
    """The free ranker fills reason/tags/summary without any API call."""
    paper = make_paper(
        title="Prescribed burning and wildfire",
        abstract="Prescribed burning lowers wildfire risk substantially over time.",
    )
    ranked = rank_papers([paper], make_config(interests="prescribed burning wildfire"))
    assert ranked[0].reason
    assert ranked[0].tags
    assert ranked[0].summary
