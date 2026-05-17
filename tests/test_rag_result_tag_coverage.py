from __future__ import annotations

from graph.rag import analyze_result_tag_coverage


def test_result_tag_coverage_counts_normalized_tags():
    summary = analyze_result_tag_coverage(
        [
            {"id": "a", "tags": ["Battery", "storage"]},
            {"id": "b", "metadata": {"tags": [{"tag": "battery"}]}},
            {"id": "c", "tags": []},
        ]
    )

    assert summary == {
        "tag_counts": {"Battery": 2, "storage": 1},
        "untagged_result_ids": ["c"],
        "dominant_tags": ["Battery"],
        "rare_tags": ["storage"],
        "coverage_ratio": 0.667,
    }


def test_result_tag_coverage_handles_empty_input():
    assert analyze_result_tag_coverage([]) == {
        "tag_counts": {},
        "untagged_result_ids": [],
        "dominant_tags": [],
        "rare_tags": [],
        "coverage_ratio": 0.0,
    }
