from graph.store.unit_markdown_image_dimension_hint_summary import summarize_unit_markdown_image_dimension_hints


def test_image_dimension_hint_summary_detects_wiki_and_markdown_width_forms():
    summary = summarize_unit_markdown_image_dimension_hints(
        [
            {"id": "b", "content": "![[file.png|300x200]] ![[icon.svg|64]]\n```md\n![[skip.png|1x2]]\n```"},
            {"id": "a", "content": "![alt](image.png =400x250) ![alt](thumb.jpg =120)"},
        ],
        sample_limit=3,
    )

    assert summary["total_units"] == 2
    assert summary["dimension_hint_count"] == 4
    assert summary["affected_units"] == 2
    assert summary["hint_counts"] == {"width_height": 2, "width_only": 2}
    assert summary["examples"] == [
        {"unit_id": "a", "line": 1, "target": "image.png", "width": "400", "height": "250"},
        {"unit_id": "a", "line": 1, "target": "thumb.jpg", "width": "120", "height": ""},
        {"unit_id": "b", "line": 1, "target": "file.png", "width": "300", "height": "200"},
    ]
