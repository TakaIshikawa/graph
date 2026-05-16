from __future__ import annotations

from types import SimpleNamespace

from graph.rag.evidence_redundancy import score_evidence_redundancy


def test_evidence_redundancy_groups_exact_duplicates():
    score = score_evidence_redundancy(
        [
            {"id": "a", "content": "Alpha beta gamma delta evidence."},
            {"id": "b", "content": "Alpha beta gamma delta evidence."},
            {"id": "c", "content": "Separate theta lambda material."},
        ]
    )

    assert score["duplicate_groups"] == [
        {"result_ids": ["a", "b"], "max_overlap": 1.0, "pair_count": 1}
    ]
    assert score["compared_count"] == 3
    assert score["distinct_count"] == 2
    assert score["redundancy_score"] == 0.3333


def test_evidence_redundancy_threshold_controls_partial_overlap():
    results = [
        {"id": "a", "content": "alpha beta gamma delta epsilon"},
        {"id": "b", "content": "alpha beta gamma delta zeta"},
    ]

    assert score_evidence_redundancy(results, overlap_threshold=0.7)[
        "duplicate_groups"
    ] == []
    assert score_evidence_redundancy(results, overlap_threshold=0.65)[
        "duplicate_groups"
    ][0]["result_ids"] == ["a", "b"]


def test_evidence_redundancy_accepts_tuple_object_and_ignores_short_content():
    result = score_evidence_redundancy(
        [
            (SimpleNamespace(id="obj", text="one two"), 0.9),
            {"id": "map", "content": "full distinct evidence words"},
        ]
    )

    assert result["duplicate_groups"] == []
    assert result["compared_count"] == 0
    assert result["distinct_count"] == 2


def test_evidence_redundancy_empty_input():
    assert score_evidence_redundancy([]) == {
        "duplicate_groups": [],
        "redundancy_score": 0.0,
        "compared_count": 0,
        "distinct_count": 0,
    }
