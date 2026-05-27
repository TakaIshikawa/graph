from __future__ import annotations

from graph.store import summarize_unit_math_notation


def test_math_notation_summary_counts_delimiters_and_unclosed():
    report = summarize_unit_math_notation([{"id": "u", "content": "$x$ $12.00$ $$y$$ \\(z\\)\n```python\n$ignored$\n```\n```math\na=b\n```\n$open"}])

    counts = {row["delimiter"]: row["count"] for row in report["delimiter_counts"]}
    assert counts == {"bracket_math": 1, "display_dollar": 1, "fenced_math": 1, "inline_dollar": 1}
    assert report["unclosed_delimiter_count"] == 1
    assert report["unclosed_examples"] == [{"unit_id": "u", "line": 8, "preview": "$open"}]
