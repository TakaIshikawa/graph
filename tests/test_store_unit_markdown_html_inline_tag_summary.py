from graph.store.unit_markdown_html_inline_tag_summary import summarize_unit_markdown_html_inline_tags


def test_html_inline_tag_summary_counts_inline_tags_and_ignores_blocks_comments_fences():
    summary = summarize_unit_markdown_html_inline_tags(
        [
            {"id": "b", "content": "Text <span>x</span> and <SUP>1</SUP>\n<div>\n<!-- <mark>x</mark> -->"},
            {"id": "a", "content": "A <a href='/'>link</a> with <mark>mark</mark>\n```html\n<sub>x</sub>\n```"},
        ],
        sample_limit=3,
    )

    assert summary["total_units"] == 2
    assert summary["inline_tag_count"] == 8
    assert summary["affected_units"] == 2
    assert summary["tag_counts"] == {"a": 2, "mark": 2, "span": 2, "sup": 2}
    assert summary["examples"] == [
        {"unit_id": "a", "line": 1, "tag": "a"},
        {"unit_id": "a", "line": 1, "tag": "a"},
        {"unit_id": "a", "line": 1, "tag": "mark"},
    ]
