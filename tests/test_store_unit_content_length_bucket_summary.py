from __future__ import annotations

from graph.store import summarize_unit_content_length_buckets


def test_unit_content_length_buckets_cover_boundaries_and_missing_content():
    summary = summarize_unit_content_length_buckets(
        [
            {"id": "empty"},
            {"id": "short-min", "content": "x"},
            {"id": "short-max", "content": "x" * 280},
            {"id": "medium-min", "content": "x" * 281},
            {"id": "medium-max", "content": "x" * 2000},
            {"id": "long-min", "content": "x" * 2001},
            {"id": "long-max", "content": "x" * 10000},
            {"id": "very-long", "content": "x" * 10001},
        ]
    )

    assert summary["bucket_counts"] == {"empty": 1, "short": 2, "medium": 2, "long": 2, "very_long": 1}
    assert {row["unit_id"]: row["bucket"] for row in summary["units"]} == {
        "empty": "empty",
        "short-min": "short",
        "short-max": "short",
        "medium-min": "medium",
        "medium-max": "medium",
        "long-min": "long",
        "long-max": "long",
        "very-long": "very_long",
    }
