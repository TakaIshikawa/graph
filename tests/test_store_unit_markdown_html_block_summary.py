from __future__ import annotations

from graph.store.unit_markdown_html_block_summary import summarize_unit_markdown_html_blocks


def test_html_block_summary_groups_block_tags_case_insensitively():
    summary = summarize_unit_markdown_html_blocks(
        [
            {"id": "u1", "content": "<DIV class=x>\nText <span>inline</span>\n<Table>"},
            {"id": "u2", "content": "```html\n<section>\n```\n<iframe src=x></iframe>\n<details>"},
        ]
    )

    assert summary["total_units"] == 2
    assert [(row["tag"], row["block_count"], row["unit_count"]) for row in summary["html_blocks"]] == [
        ("details", 1, 1),
        ("div", 1, 1),
        ("iframe", 1, 1),
        ("table", 1, 1),
    ]
    assert summary["html_blocks"][1]["examples"][0]["snippet"] == "<DIV class=x>"
