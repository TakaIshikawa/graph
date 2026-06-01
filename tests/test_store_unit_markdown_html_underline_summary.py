from __future__ import annotations

from graph.store import summarize_unit_markdown_html_underlines


def test_markdown_html_underline_summary_extracts_same_line_spans_only():
    report = summarize_unit_markdown_html_underlines(
        [
            {"id": "u2", "content": "<U>Important</U>\n<u>open\n```html\n<u>Hidden</u>\n```"},
            {"id": "u1", "content": "See <u>Important</u> and <u>Other</u>."},
        ]
    )

    assert report["total_units"] == 2
    assert report["units_with_underline"] == 2
    assert report["underline_count"] == 3
    assert report["most_common_text"] == "Important"
    assert report["samples"] == [
        {"unit_id": "u1", "line_number": 1, "text": "Important"},
        {"unit_id": "u1", "line_number": 1, "text": "Other"},
        {"unit_id": "u2", "line_number": 1, "text": "Important"},
    ]
