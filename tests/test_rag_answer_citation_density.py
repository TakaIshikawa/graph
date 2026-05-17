from __future__ import annotations

import pytest

from graph.rag import estimate_answer_citation_density


def test_answer_citation_density_counts_brackets_and_markdown_links():
    summary = estimate_answer_citation_density(
        "Claim one [1].\n\nClaim two without citation.\n\nClaim three [2] and [source](https://example.test).",
        max_citations_per_paragraph=1,
    )

    assert summary == {
        "paragraph_count": 3,
        "citation_count": 3,
        "uncited_paragraph_indexes": [1],
        "over_cited_paragraph_indexes": [2],
        "density_score": 0.333,
    }


def test_answer_citation_density_handles_empty_answers():
    assert estimate_answer_citation_density("") == {
        "paragraph_count": 0,
        "citation_count": 0,
        "uncited_paragraph_indexes": [],
        "over_cited_paragraph_indexes": [],
        "density_score": 0.0,
    }


def test_answer_citation_density_validates_thresholds():
    with pytest.raises(ValueError, match="thresholds"):
        estimate_answer_citation_density("Answer [1].", min_citations_per_paragraph=2, max_citations_per_paragraph=1)
