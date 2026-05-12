from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.source_balance import score_source_balance


@dataclass
class Result:
    source_project: str


def test_source_balance_scores_dominant_source_and_warnings():
    summary = score_source_balance(
        [
            {"source_project": "notes"},
            {"metadata": {"source_project": "notes"}},
            Result("web"),
            {"title": "missing"},
        ],
        ideal_min_sources=3,
    )

    assert summary["total_results"] == 4
    assert summary["source_count"] == 3
    assert summary["dominant_source"] == "notes"
    assert summary["dominant_ratio"] == 0.5
    assert summary["source_counts"] == {"notes": 2, "unknown": 1, "web": 1}
    assert summary["warnings"] == ["unknown_source_project"]


def test_source_balance_handles_empty_results():
    assert score_source_balance([]) == {
        "total_results": 0,
        "source_count": 0,
        "dominant_source": None,
        "dominant_ratio": 0,
        "balance_score": 0,
        "source_counts": {},
        "warnings": ["no_results"],
    }


def test_source_balance_warns_for_too_few_and_dominant_sources():
    summary = score_source_balance([{"source_project": "a"}, {"source_project": "a"}, {"source_project": "b"}])

    assert summary["warnings"] == ["too_few_sources", "dominant_source_concentration"]


@pytest.mark.parametrize("ideal_min_sources", [0, -1, 1.2, True, "3"])
def test_source_balance_validates_ideal_min_sources(ideal_min_sources):
    with pytest.raises(ValueError, match="ideal_min_sources must be a positive integer"):
        score_source_balance([], ideal_min_sources=ideal_min_sources)
