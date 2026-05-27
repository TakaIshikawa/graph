from __future__ import annotations

from graph.store.unit_markdown_table_shape_summary import summarize_unit_markdown_table_shapes


def test_markdown_table_shape_summary_reports_shapes_buckets_and_malformed_units():
    report = summarize_unit_markdown_table_shapes(
        [
            {"id": "a", "content": "| A | B | C |\n| - | - | - |\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |"},
            {"id": "b", "content": "| A | B |\n| nope | - |\n| 1 | 2 |"},
            {"id": "c", "content": "Prose with A | B but no table."},
        ]
    )

    assert report["total_tables"] == 1
    assert report["table_shapes"] == [{"unit_id": "a", "rows": 2, "columns": 3}]
    assert report["row_bucket_counts"] == [
        {"bucket": "0", "count": 0},
        {"bucket": "1-2", "count": 1},
        {"bucket": "3-5", "count": 0},
        {"bucket": "6+", "count": 0},
    ]
    assert report["column_counts"] == [{"columns": 3, "count": 1}]
    assert report["malformed_units"] == ["b"]
