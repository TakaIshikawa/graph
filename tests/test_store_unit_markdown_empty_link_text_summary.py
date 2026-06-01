from graph.store.unit_markdown_empty_link_text_summary import summarize_unit_markdown_empty_link_texts


def test_empty_link_text_summary_counts_links_images_and_ignores_fences():
    summary = summarize_unit_markdown_empty_link_texts(
        [
            {"id": "b", "content": "[](b.md)\n![](img.png)\n```md\n[](skip.md)\n![](skip.png)\n```"},
            {"id": "a", "content": "[](a.md) [ok](x) ![alt](y)"},
        ],
        sample_limit=2,
    )

    assert summary["total_units"] == 2
    assert summary["links_with_empty_text"] == 2
    assert summary["images_with_empty_alt"] == 1
    assert summary["affected_units"] == 2
    assert summary["examples"] == [
        {"unit_id": "a", "line": 1, "target": "a.md", "is_image": False},
        {"unit_id": "b", "line": 1, "target": "b.md", "is_image": False},
    ]


def test_empty_link_text_summary_allows_zero_sample_limit():
    summary = summarize_unit_markdown_empty_link_texts([{"id": "u", "content": "[](target) ![](image.png)"}], sample_limit=0)

    assert summary["links_with_empty_text"] == 1
    assert summary["images_with_empty_alt"] == 1
    assert summary["examples"] == []
