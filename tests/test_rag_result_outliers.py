from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag.result_outliers import detect_result_outliers


@dataclass
class Result:
    id: str
    title: str
    content: str
    tags: list[str]


def test_detect_result_outliers_flags_low_overlap_results():
    report = detect_result_outliers(
        [
            {"id": "a", "title": "Solar storage", "content": "Battery grid storage planning"},
            {"id": "b", "title": "Grid battery", "content": "Solar storage deployment"},
            {"id": "c", "title": "Sourdough starter", "content": "Flour hydration fermentation"},
        ],
        min_overlap=0.25,
    )

    assert report["baseline_terms"] == ["battery", "grid", "solar", "storage"]
    assert report["outliers"] == [
        {
            "result_id": "c",
            "overlap_score": 0.0,
            "shared_terms": [],
            "distinctive_terms": ["fermentation", "flour", "hydration", "sourdough", "starter"],
            "reason": "low token overlap with retrieved result set",
        }
    ]


def test_detect_result_outliers_supports_objects_tuple_wrappers_and_metadata():
    report = detect_result_outliers(
        [
            (Result("obj", "Alpha plan", "Shared roadmap", ["team"]), 0.8),
            {"metadata": {"unit_id": "meta", "title": "Alpha roadmap", "tags": ["team", "shared"]}},
            {"id": "odd", "content": "zebra quartz"},
        ],
        min_overlap=0.2,
    )

    assert [item["result_id"] for item in report["outliers"]] == ["odd"]
    assert "alpha" in report["baseline_terms"]


def test_detect_result_outliers_ignores_common_stop_words():
    report = detect_result_outliers(
        [
            {"id": "a", "content": "the and of solar"},
            {"id": "b", "content": "the and of wind"},
        ]
    )

    assert "the" not in report["token_frequencies"]
    assert report["baseline_terms"] == []


@pytest.mark.parametrize("min_overlap", [-0.1, True, "0.2"])
def test_detect_result_outliers_validates_min_overlap(min_overlap):
    with pytest.raises(ValueError, match="min_overlap must be a non-negative number"):
        detect_result_outliers([], min_overlap=min_overlap)
