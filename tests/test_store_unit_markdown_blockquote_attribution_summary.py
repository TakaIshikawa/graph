from graph.store import summarize_unit_markdown_blockquote_attributions


def test_blockquote_attribution_summary_groups_contiguous_quotes():
    content = "> Quote line\n> -- Ada\n\n> Unattributed"

    result = summarize_unit_markdown_blockquote_attributions([{"id": "u", "content": content}])

    assert result["total_blockquotes"] == 2
    assert result["attributed_count"] == 1
    assert result["unattributed_count"] == 1
    assert result["attribution_counts"] == [{"attribution": "Ada", "count": 1}]
    assert result["samples"][0]["line_range"] == "1-2"
