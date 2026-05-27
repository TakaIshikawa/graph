from graph.store import summarize_unit_markdown_unicode_emoji


def test_unicode_emoji_summary_counts_unicode_not_shortcodes_or_fences():
    summary = summarize_unit_markdown_unicode_emoji([{"id": "u1", "content": "Hi 😀 😀 :sparkles:\n```\n✅\n```"}])

    assert summary["units_with_emoji"] == 1
    assert summary["total_emoji"] == 2
    assert summary["emoji_frequency"] == [{"emoji": "😀", "count": 2}]
