from __future__ import annotations

from graph.store.unit_markdown_escape_summary import summarize_unit_markdown_escapes


def test_escape_summary_counts_escapes_dangling_and_common_character():
    summary = summarize_unit_markdown_escapes([
        {"id": "u1", "source": "s", "content": r"\* item \[x\]"},
        {"id": "u2", "source": "s", "content": "dangling\\\n```\n\\* ignored\n```"},
    ])

    assert summary["sources"] == [
        {"source": "s", "unit_count": 2, "units_with_escapes": 2, "escape_sequence_count": 3, "dangling_backslash_count": 1, "most_common_escaped_character": "*"}
    ]
