from __future__ import annotations

from graph.store import summarize_unit_markdown_blockquote_depths


def test_markdown_blockquote_depth_summary_counts_nested_lines_outside_fences():
    report = summarize_unit_markdown_blockquote_depths(
        [
            {"id": "b", "content": "> one\n>> two\n```md\n>>> ignored\n```"},
            {"id": "a", "content": ">>> three\nplain"},
        ]
    )

    assert report["total_units"] == 2
    assert report["quote_line_count"] == 3
    assert report["max_depth"] == 3
    assert report["depth_counts"] == [{"depth": 1, "count": 1}, {"depth": 2, "count": 1}, {"depth": 3, "count": 1}]
    assert report["units_with_nested_blockquotes"] == 2
    assert report["samples"] == [
        {"unit_id": "a", "line_number": 1, "depth": 3, "text": "three"},
        {"unit_id": "b", "line_number": 2, "depth": 2, "text": "two"},
    ]
