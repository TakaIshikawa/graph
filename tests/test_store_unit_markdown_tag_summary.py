from graph.store import summarize_unit_markdown_tags


def test_summary_counts_standalone_and_nested_body_tags():
    summary = summarize_unit_markdown_tags([{"content": "Body #Topic #Topic/Sub"}, {"content": "# Heading\n`#code`\nhttps://x.test/#frag\n#Topic"}])
    assert summary["total_tags"] == 3
    assert summary["units_with_tags"] == 2
    assert summary["tag_counts"] == {"#topic": 2, "#topic/sub": 1}
    assert summary["nested_tag_counts"] == {"#topic/sub": 1}
    assert summary["max_tag_depth"] == 2
