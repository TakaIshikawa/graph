from graph.store import summarize_unit_markdown_inline_code


def test_summary_counts_single_and_multi_backtick_spans():
    summary = summarize_unit_markdown_inline_code([{"content": "`x` ``y ` z`` `x`"}])
    assert summary["total_spans"] == 3
    assert summary["units_with_inline_code"] == 1
    assert summary["delimiter_length_counts"] == {1: 2, 2: 1}
    assert summary["most_common_code_spans"][0] == {"code": "x", "count": 2}


def test_summary_ignores_fences_and_unclosed_delimiters():
    summary = summarize_unit_markdown_inline_code([{"content": "```\n`skip`\n```\n`ok`\n`open"}])
    assert summary["total_spans"] == 1
    assert summary["average_code_length"] == 2
