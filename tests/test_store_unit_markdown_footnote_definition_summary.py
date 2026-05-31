from graph.store import summarize_unit_markdown_footnote_definitions


def test_footnote_definition_summary_counts_duplicates_and_multiline():
    content = "[^a]: One\n    more\n[^a]: Two\n[^b]: Three"

    result = summarize_unit_markdown_footnote_definitions([{"id": "u", "content": content}])

    assert result["total_definitions"] == 3
    assert result["units_with_definitions"] == 1
    assert result["duplicate_labels"] == [{"unit_id": "u", "label": "a", "count": 2}]
    assert result["multiline_count"] == 1
