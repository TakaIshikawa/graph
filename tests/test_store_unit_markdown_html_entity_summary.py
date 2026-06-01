from graph.store.unit_markdown_html_entity_summary import summarize_unit_markdown_html_entities


def test_summarizes_named_decimal_and_hex_entities_ignoring_plain_ampersands():
    report = summarize_unit_markdown_html_entities([
        {"id": "b", "content": "&amp; &amp; &copy; &not-an-entity; &"},
        {"id": "a", "content": "&#169; &#x1F600; &#xBAD; &#bad;"},
    ])

    assert report["total_entity_count"] == 6
    assert report["entity_counts"] == {"&#169;": 1, "&#x1F600;": 1, "&#xBAD;": 1, "&amp;": 2, "&copy;": 1}
    assert report["units"] == [
        {"unit_id": "a", "entity_count": 3, "entity_counts": {"&#169;": 1, "&#x1F600;": 1, "&#xBAD;": 1}, "top_entity": "&#169;"},
        {"unit_id": "b", "entity_count": 3, "entity_counts": {"&amp;": 2, "&copy;": 1}, "top_entity": "&amp;"},
    ]
