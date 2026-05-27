from __future__ import annotations

from graph.store import summarize_unit_blockquote_usage


def test_summarize_unit_blockquote_usage_groups_blocks_and_density():
    summary = summarize_unit_blockquote_usage([{"id": "a", "content": "> one\n> two\nplain\n> three"}, {"id": "b", "content": "plain"}])

    assert summary["units_with_blockquotes"] == 1
    assert summary["total_quote_blocks"] == 2
    assert summary["total_quoted_lines"] == 3
    assert summary["quote_density_buckets"] == {"high": 1, "none": 1}
    assert summary["top_units_by_quoted_lines"][0]["unit_id"] == "a"
