from __future__ import annotations

from graph.store.unit_markdown_html_mark_summary import summarize_unit_markdown_html_marks


def test_detects_mark_tags_and_attributes_outside_fences():
    summary = summarize_unit_markdown_html_marks(
        [
            {
                "id": "u1",
                "content": '<mark class="hot" data-kind="risk">Important</mark> and <mark>Plain</mark>\n```\n<mark class="x">No</mark>\n```',
            }
        ]
    )

    assert summary["mark_count"] == 2
    assert summary["units_with_marks"] == 1
    assert summary["attribute_key_counts"] == {"class": 1, "data-kind": 1}
    assert summary["samples"] == [
        {"unit_id": "u1", "line_number": 1, "text": "Important", "attributes": ["class", "data-kind"]},
        {"unit_id": "u1", "line_number": 1, "text": "Plain", "attributes": []},
    ]
