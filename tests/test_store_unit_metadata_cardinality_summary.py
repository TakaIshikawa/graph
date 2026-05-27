from __future__ import annotations

from graph.store import summarize_unit_metadata_cardinality


def test_metadata_cardinality_flattens_lists_counts_blanks_and_repeats():
    summary = summarize_unit_metadata_cardinality(
        [
            {"id": "a", "metadata": {"status": "open", "tags": ["ai", "ops"], "owner": ""}},
            {"id": "b", "metadata": {"status": "open", "tags": ["ai"], "owner": None}},
            {"id": "c", "metadata": {"status": "closed", "tags": [], "owner": "Taka"}},
        ]
    )

    rows = {row["key"]: row for row in summary["rows"]}
    assert rows["status"]["distinct_value_count"] == 2
    assert rows["status"]["repeated_value_count"] == 2
    assert rows["tags"]["frequent_values"][0] == {"value": "ai", "count": 2}
    assert rows["tags"]["blank_value_count"] == 1
    assert rows["owner"]["blank_value_count"] == 2
