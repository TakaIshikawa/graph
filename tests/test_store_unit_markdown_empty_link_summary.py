from graph.store import summarize_unit_markdown_empty_links


def test_empty_link_summary_detects_empty_whitespace_and_anchor_placeholder():
    result = summarize_unit_markdown_empty_links([{"id": "u", "content": "[empty]() [space](   ) [anchor](#) [ok](/x) ![img]()\n```\n[hidden]()\n```"}])

    assert result["total_units"] == 1
    assert result["units_with_empty_links"] == 1
    assert result["empty_link_count"] == 3
    assert result["anchor_placeholder_count"] == 1
    assert [sample["label"] for sample in result["samples"]] == ["empty", "space", "anchor"]
