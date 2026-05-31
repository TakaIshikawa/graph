from graph.store import summarize_unit_markdown_table_captions


def test_table_caption_summary_counts_preceding_and_uncaptioned_tables():
    content = "Table: Metrics\n| A | B |\n| - | - |\n| 1 | 2 |\n\n| C | D |\n| - | - |"

    result = summarize_unit_markdown_table_captions([{"id": "u", "content": content}])

    assert result["total_tables"] == 2
    assert result["captioned_tables"] == 1
    assert result["uncaptioned_tables"] == 1
    assert result["caption_position_counts"] == [{"position": "preceding", "count": 1}]
