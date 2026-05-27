from __future__ import annotations

from graph.store.unit_markdown_highlight_summary import summarize_unit_markdown_highlights


def test_highlight_summary_groups_counts_and_average_lengths():
    summary = summarize_unit_markdown_highlights([
        {"id": "u1", "source": "s", "content": "==one== and ==three== `==code==`"},
        {"id": "u2", "source": "s", "content": "```\n==ignored==\n```\n== =="},
    ])

    assert summary["sources"] == [
        {"source": "s", "unit_count": 2, "units_with_highlights": 2, "highlight_span_count": 3, "max_highlights_per_unit": 2, "average_highlight_text_length": 2.67}
    ]
