from __future__ import annotations

import pytest

from graph.rag.citation_diversity import analyze_citation_diversity


def test_citation_diversity_computes_metadata_ratios_and_dominant_groups():
    report = analyze_citation_diversity(
        [
            {"id": "a", "url": "https://one.test/a", "author": "Ada", "source_type": "paper", "year": 2025},
            {"id": "b", "url": "https://two.test/b", "author": "Ben", "source_type": "paper", "date": "2026-01-01"},
            {"id": "c", "url": "https://two.test/c", "author": "Ben", "source_type": "blog", "date": "2026-02-01"},
        ],
        dominance_threshold=0.66,
    )

    assert report["metrics"]["domain"]["unique_count"] == 2
    assert report["metrics"]["domain"]["diversity_ratio"] == 0.667
    assert report["dominant_groups"]["author"] == {"value": "ben", "count": 2, "ratio": 0.667}
    assert "dominant_author:ben" in report["warnings"]
    assert report["citations"][0]["citation_id"] == "a"


def test_citation_diversity_handles_empty_and_validates_threshold():
    assert analyze_citation_diversity([])["metrics"]["domain"]["diversity_ratio"] == 0.0
    with pytest.raises(ValueError):
        analyze_citation_diversity([], dominance_threshold=0)
