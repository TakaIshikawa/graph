from graph.store import summarize_unit_markdown_link_fragments


def test_link_fragment_summary_counts_internal_external_and_ignores_images_fences():
    content = "[a](#one) [b](doc.md#one) [c](https://e.test/x#two) ![i](x#bad)\n```\n[h](x#bad)\n```"
    summary = summarize_unit_markdown_link_fragments([{"id": "u1", "content": content}])

    assert summary["total_fragment_links"] == 3
    assert summary["internal_fragment_count"] == 2
    assert summary["external_fragment_count"] == 1
    assert summary["top_fragments"][0] == {"fragment": "one", "count": 2}
