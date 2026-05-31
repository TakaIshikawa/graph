from __future__ import annotations

from graph.store.unit_frontmatter_date_range_summary import summarize_unit_frontmatter_date_ranges


def test_frontmatter_date_range_summary_counts_complete_missing_and_inverted():
    summary = summarize_unit_frontmatter_date_ranges(
        [
            {"id": "a", "content": "---\nstart: 2024-01-01\nend: 2024-01-31\n---\nBody"},
            {"id": "b", "metadata": {"start_date": "2024-03-01"}},
            {"id": "c", "content": "---\nfrom: 2024-05-01\nto: 2024-04-01\n---"},
            {"id": "d", "metadata": {"valid_to": "2024-06-01"}},
        ]
    )

    by_pair = {row["field_pair"]: row for row in summary["date_ranges"]}
    assert by_pair["start/end"]["complete"] == 1
    assert by_pair["start_date/end_date"]["missing_end"] == 1
    assert by_pair["from/to"]["complete"] == 1
    assert by_pair["from/to"]["inverted"] == 1
    assert by_pair["valid_from/valid_to"]["missing_start"] == 1
    assert by_pair["valid_from/valid_to"]["examples"][0]["unit_id"] == "d"
