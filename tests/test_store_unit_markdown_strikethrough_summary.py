from __future__ import annotations

from graph.store.unit_markdown_strikethrough_summary import summarize_unit_markdown_strikethrough


def test_strikethrough_summary_groups_by_source_and_ignores_code():
    summary = summarize_unit_markdown_strikethrough([
        {"id": "u2", "source": "b", "content": "`~~code~~`\n~~one~~\n~~two~~"},
        {"id": "u1", "source": "a", "content": "```md\n~~ignored~~\n```\n~~kept~~"},
        {"id": "u3", "source": "a", "content": "none"},
    ])

    assert summary["sources"] == [
        {"source": "a", "unit_count": 2, "units_with_strikethrough": 1, "strikethrough_span_count": 1, "max_spans_per_unit": 1, "sample_units": ["u1"]},
        {"source": "b", "unit_count": 1, "units_with_strikethrough": 1, "strikethrough_span_count": 2, "max_spans_per_unit": 2, "sample_units": ["u2"]},
    ]
