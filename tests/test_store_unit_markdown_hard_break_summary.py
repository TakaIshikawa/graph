from __future__ import annotations

from graph.store.unit_markdown_hard_break_summary import summarize_unit_markdown_hard_breaks


def test_hard_break_summary_distinguishes_break_types_and_ignores_fences():
    summary = summarize_unit_markdown_hard_breaks([
        {"id": "u1", "source": "s", "content": "space  \nnext\\\nend"},
        {"id": "u2", "source": "s", "content": "```\ncode  \ncode\\\n```\nplain"},
    ])

    assert summary["sources"] == [
        {"source": "s", "unit_count": 2, "units_with_hard_breaks": 1, "hard_break_count": 2, "trailing_space_break_count": 1, "backslash_break_count": 1, "max_hard_breaks_per_unit": 2}
    ]
