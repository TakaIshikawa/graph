from graph.store import summarize_unit_markdown_math_spans


def test_math_span_summary_counts_inline_and_block_math_ignoring_escaped_dollars():
    content = "Inline $x+1$ and escaped \\$no$.\n$$\ny=2\n$$"

    result = summarize_unit_markdown_math_spans([{"id": "u", "content": content}])

    assert result["total_math_spans"] == 2
    assert result["units_with_math"] == 1
    assert result["inline_count"] == 1
    assert result["block_count"] == 1
