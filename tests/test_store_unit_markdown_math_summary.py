from __future__ import annotations

from graph.store import summarize_unit_markdown_math


def test_markdown_math_summary_counts_inline_block_and_examples():
    report = summarize_unit_markdown_math([
        {"id": "u1", "title": "One", "content": "Inline $x+y$ and price $12.00$.\n$$a=b$$"},
        {"id": "u2", "content": "```math\nc=d\n```\n`$ignored$`"},
    ])

    assert report["total_expression_count"] == 3
    assert report["inline_expression_count"] == 1
    assert report["block_expression_count"] == 2
    assert report["units_containing_math"] == 2
    assert report["examples"][0]["unit_id"] == "u1"
