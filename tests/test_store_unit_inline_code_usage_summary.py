from __future__ import annotations

from graph.store import summarize_unit_inline_code_usage


def test_summarize_unit_inline_code_usage_ignores_fenced_code_and_counts_snippets():
    summary = summarize_unit_inline_code_usage([{"id": "a", "content": "Use `foo` and `foo`.\n```\n`bar`\n```"}, {"id": "b", "content": "plain"}])

    assert summary["inline_code_span_count"] == 2
    assert summary["distinct_code_tokens"] == 1
    assert summary["common_snippets"] == [{"snippet": "foo", "count": 2}]
    assert summary["high_density_units"][0]["unit_id"] == "a"
