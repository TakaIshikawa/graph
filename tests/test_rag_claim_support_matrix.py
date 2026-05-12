from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag import build_claim_support_matrix


@dataclass
class ResultStub:
    id: str
    source_project: str
    title: str
    content: str
    tags: list[str]


def test_build_claim_support_matrix_matches_normalized_non_stopword_overlap():
    rows = build_claim_support_matrix(
        [
            {
                "id": "r1",
                "source_project": "alpha",
                "title": "Solar storage plan",
                "content": "Grid batteries improve resilience.",
                "tags": ["Energy"],
            },
            {
                "id": "r2",
                "source_project": "beta",
                "title": "Finance note",
                "content": "Procurement costs delay rollout.",
                "tags": ["budget"],
            },
        ],
        ["The solar storage plan improve grid resilience"],
        min_overlap=2,
    )

    assert rows == [
        {
            "claim": "The solar storage plan improve grid resilience",
            "supporting_result_ids": ["r1"],
            "source_projects": ["alpha"],
            "matched_terms": ["grid", "improve", "plan", "resilience", "solar", "storage"],
            "support_count": 1,
            "support_level": "weak",
        }
    ]


def test_build_claim_support_matrix_supports_objects_mappings_snippets_and_tags():
    rows = build_claim_support_matrix(
        [
            ResultStub(
                id="object-1",
                source_project="object-source",
                title="Reliability memo",
                content="Storage supports reliability",
                tags=["grid planning"],
            ),
            {
                "unit": {
                    "id": "nested-1",
                    "title": "Operations",
                    "content": "brief",
                    "tags": [{"tag": "load forecasting"}],
                    "metadata": {"source_project": "nested-source"},
                },
                "metadata": {"snippet": "Forecasting reduces peak load risk."},
            },
        ],
        [
            "Reliability improves with storage",
            "Peak load forecasting risk is visible",
        ],
    )

    assert rows[0]["supporting_result_ids"] == ["object-1"]
    assert rows[0]["source_projects"] == ["object-source"]
    assert rows[0]["matched_terms"] == ["reliability", "storage"]
    assert rows[1]["supporting_result_ids"] == ["nested-1"]
    assert rows[1]["source_projects"] == ["nested-source"]
    assert rows[1]["matched_terms"] == ["forecasting", "load", "peak", "risk"]


def test_build_claim_support_matrix_distinguishes_none_weak_and_strong_support():
    results = [
        {"id": "b", "source_project": "beta", "title": "Solar finance", "content": ""},
        {"id": "a", "source_project": "alpha", "title": "Solar storage", "content": ""},
        {"id": "c", "source_project": "alpha", "title": "Wind operations", "content": ""},
    ]

    rows = build_claim_support_matrix(
        results,
        [
            "Solar deployment",
            "Wind maintenance",
            "Nuclear roadmap",
        ],
    )

    assert [row["support_level"] for row in rows] == ["strong", "weak", "none"]
    assert rows[0]["supporting_result_ids"] == ["a", "b"]
    assert rows[0]["source_projects"] == ["alpha", "beta"]
    assert rows[2]["matched_terms"] == []


def test_build_claim_support_matrix_is_deterministic_across_result_order():
    results = [
        {"id": "b", "source_project": "beta", "title": "Grid storage", "content": ""},
        {"id": "a", "source_project": "alpha", "title": "Grid storage", "content": ""},
    ]

    assert build_claim_support_matrix(results, ["grid storage"]) == build_claim_support_matrix(
        reversed(results),
        ["grid storage"],
    )


@pytest.mark.parametrize("min_overlap", [0, -1, "1", True])
def test_build_claim_support_matrix_validates_min_overlap(min_overlap):
    with pytest.raises(ValueError, match="min_overlap must be a positive integer"):
        build_claim_support_matrix([], [], min_overlap=min_overlap)


def test_build_claim_support_matrix_is_importable_from_graph_rag():
    assert callable(build_claim_support_matrix)
