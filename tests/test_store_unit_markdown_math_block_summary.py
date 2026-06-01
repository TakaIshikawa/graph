from graph.store.unit_markdown_math_block_summary import summarize_unit_markdown_math_blocks


def test_summarizes_complete_multiple_and_unterminated_math_blocks():
    report = summarize_unit_markdown_math_blocks([
        {"id": "b", "content": "$ inline $\n$$\na\nb\n$$\n$$\nc\n$$"},
        {"id": "a", "content": "$$\nx\n```\n$$ ignored fence toggle while in math"},
    ])

    assert report["total_math_blocks"] == 3
    assert report["unterminated_block_count"] == 1
    assert report["units"] == [
        {"unit_id": "a", "math_block_count": 1, "first_line_number": 1, "max_block_line_count": 3, "unterminated_block_count": 1},
        {"unit_id": "b", "math_block_count": 2, "first_line_number": 2, "max_block_line_count": 2, "unterminated_block_count": 0},
    ]
