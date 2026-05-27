import pytest

from graph.rag.result_snippet_overlap import audit_result_snippet_overlap


def test_result_snippet_overlap_reports_pairs_above_threshold():
    report = audit_result_snippet_overlap(
        [
            {"id": "a", "snippet": "Solar storage battery policy incentives"},
            {"id": "b", "snippet": "Solar storage battery incentives"},
            {"id": "c", "snippet": "Marine biology habitat survey"},
        ],
        threshold=0.5,
    )

    assert report["result_count"] == 3
    assert report["overlapping_pair_count"] == 1
    assert report["max_overlap_ratio"] == 0.8
    assert report["overlapping_pairs"] == [{"left_id": "a", "right_id": "b", "overlap_ratio": 0.8}]


def test_result_snippet_overlap_validates_threshold():
    with pytest.raises(ValueError):
        audit_result_snippet_overlap([], threshold=1.1)
