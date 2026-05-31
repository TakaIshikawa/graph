from __future__ import annotations

from graph.store.unit_markdown_emoji_shortcode_summary import summarize_unit_markdown_emoji_shortcodes


def test_markdown_emoji_shortcode_summary_counts_valid_and_avoids_false_positives():
    summary = summarize_unit_markdown_emoji_shortcodes(
        [
            {"id": "u1", "content": ":warning: :book-open: :custom_name:\nhttps://example.com/a:b\n`:code:`"},
            {"id": "u2", "content": ":warning:\n```\n:book:\n```\nBad :no space: and :!:"},
        ]
    )

    assert summary["shortcodes"] == [
        {"shortcode": "warning", "count": 2, "unit_count": 2},
        {"shortcode": "book-open", "count": 1, "unit_count": 1},
        {"shortcode": "custom_name", "count": 1, "unit_count": 1},
    ]
    assert summary["units_with_shortcodes"] == 2
    assert {row["token"] for row in summary["malformed_tokens"]} == {":!:"}
