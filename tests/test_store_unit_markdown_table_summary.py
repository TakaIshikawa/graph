from __future__ import annotations

from graph.store.unit_markdown_table_summary import summarize_unit_markdown_tables


def test_markdown_table_summary_counts_valid_tables_and_malformed_candidates():
    report = summarize_unit_markdown_tables(
        [
            {"id": "a", "content": "| A | B |\n| - | - |\n| 1 | 2 |\n| 3 | 4 |"},
            {"id": "b", "content": "| A | B |\n| bad | - |\n| 1 | 2 |"},
            {"id": "c", "content": "This prose has A | B only."},
        ]
    )

    assert report["total_units"] == 3
    assert report["total_tables"] == 1
    assert report["units_with_tables"] == 1
    assert report["row_count_distribution"] == [{"rows": 2, "count": 1}]
    assert report["malformed_units"] == ["b"]
