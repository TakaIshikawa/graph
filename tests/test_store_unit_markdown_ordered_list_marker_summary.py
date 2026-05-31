from graph.store import summarize_unit_markdown_ordered_list_markers


def test_ordered_list_marker_summary_counts_delimiters_and_non_one_markers():
    result = summarize_unit_markdown_ordered_list_markers([{"id": "u", "content": "1. One\n2) Two\n```\n3. Hidden\n```\n10. Ten"}])

    assert result["total_units"] == 1
    assert result["units_with_ordered_lists"] == 1
    assert result["item_count"] == 3
    assert result["non_one_start_count"] == 2
    assert result["paren_delimiter_count"] == 1
    assert result["dot_delimiter_count"] == 2
    assert result["samples"][1]["delimiter"] == ")"
