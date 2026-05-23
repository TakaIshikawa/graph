from __future__ import annotations

from graph.rag import audit_answer_citation_anchors


def test_audit_answer_citation_anchors_matches_numeric_markers_to_positions():
    audit = audit_answer_citation_anchors(
        "First claim [1]. Second claim [2].",
        [{"id": "alpha"}, {"id": "beta"}],
    )

    assert audit == {
        "known_anchors": ["alpha", "beta"],
        "used_anchors": ["alpha", "beta"],
        "missing_anchors": [],
        "unknown_anchors": [],
        "anchor_coverage": 1.0,
    }


def test_audit_answer_citation_anchors_supports_named_markers_and_dict_ids():
    audit = audit_answer_citation_anchors(
        "The report says so [Smith 2024] and repeats it [Smith 2024].",
        [
            {"id": "smith-2024", "label": "Smith 2024"},
            {"id": "jones-2023", "label": "Jones 2023"},
        ],
    )

    assert audit["known_anchors"] == ["smith-2024", "jones-2023"]
    assert audit["used_anchors"] == ["smith-2024"]
    assert audit["missing_anchors"] == ["jones-2023"]
    assert audit["unknown_anchors"] == []
    assert audit["anchor_coverage"] == 0.5


def test_audit_answer_citation_anchors_reports_unknown_markers_once():
    audit = audit_answer_citation_anchors(
        "Known [alpha]. Unknown [missing] and again [missing].",
        [{"id": "alpha"}],
    )

    assert audit == {
        "known_anchors": ["alpha"],
        "used_anchors": ["alpha"],
        "missing_anchors": [],
        "unknown_anchors": ["missing"],
        "anchor_coverage": 1.0,
    }


def test_audit_answer_citation_anchors_handles_numeric_marker_lists_without_double_counting():
    audit = audit_answer_citation_anchors(
        "Both claims use the same bundle [1, 2] and repeat [1].",
        [{"id": "a"}, {"id": "b"}, {"id": "c"}],
    )

    assert audit["used_anchors"] == ["a", "b"]
    assert audit["missing_anchors"] == ["c"]
    assert audit["anchor_coverage"] == 0.667
