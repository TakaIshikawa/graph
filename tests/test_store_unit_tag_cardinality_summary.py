from __future__ import annotations

from graph.store.unit_tag_cardinality_summary import summarize_unit_tag_cardinality


def test_unit_tag_cardinality_summary_counts_lists_strings_and_buckets():
    summary = summarize_unit_tag_cardinality(
        [
            {"id": "a", "metadata": {"tags": "one, two, "}},
            {"id": "b", "tags": ["x", " ", "y", "z", "q", "r", "s"]},
            {"id": "c"},
        ]
    )

    assert summary == {
        "total_units": 3,
        "min": 0,
        "max": 6,
        "average": 2.67,
        "median": 2,
        "zero_tag_units": 1,
        "bucket_counts": {"zero": 1, "one": 0, "two_to_five": 1, "six_to_ten": 1, "over_ten": 0},
    }
