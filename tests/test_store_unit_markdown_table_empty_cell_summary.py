from __future__ import annotations

from graph.store import summarize_unit_markdown_table_empty_cells


def test_markdown_table_empty_cell_summary_counts_rows_outside_fences():
    report = summarize_unit_markdown_table_empty_cells(
        [
            {"id": "b", "content": "| A | B | C |\n| - | - | - |\n| 1 |  | 3 |\n```md\n| x |  |\n```"},
            {"id": "a", "content": "| A | B |\n| - | - |\n|  | 2 |\n| 3 |  |"},
        ]
    )

    assert report["total_units"] == 2
    assert report["units_with_empty_cells"] == 2
    assert report["total_empty_cells"] == 3
    assert report["most_common_column_position"] == 2
    assert report["samples"] == [
        {"unit_id": "a", "line_number": 3, "column_position": 1, "row_text": "|  | 2 |"},
        {"unit_id": "a", "line_number": 4, "column_position": 2, "row_text": "| 3 |  |"},
        {"unit_id": "b", "line_number": 3, "column_position": 2, "row_text": "| 1 |  | 3 |"},
    ]
