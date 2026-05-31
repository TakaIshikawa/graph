from __future__ import annotations

from graph.store.unit_markdown_table_header_summary import summarize_unit_markdown_table_headers


def test_table_header_summary_groups_header_sets_and_names_ignoring_fences():
    summary = summarize_unit_markdown_table_headers(
        [
            {"id": "u1", "content": "| Name | Status |\n| --- | --- |\n| A | B |"},
            {"id": "u2", "content": "```\n| Fake | Header |\n| --- | --- |\n```\n| Name | Owner |\n| --- | --- |"},
        ]
    )

    assert summary["table_count"] == 2
    assert summary["header_names"][0] == {"header": "name", "count": 2, "unit_count": 2}
    assert summary["header_sets"] == [
        {"headers": ["name", "owner"], "table_count": 1, "unit_count": 1, "examples": [{"unit_id": "u2", "line_number": 5, "headers": ["name", "owner"]}]},
        {"headers": ["name", "status"], "table_count": 1, "unit_count": 1, "examples": [{"unit_id": "u1", "line_number": 1, "headers": ["name", "status"]}]},
    ]
