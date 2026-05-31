from __future__ import annotations

from graph.store.unit_markdown_image_title_summary import summarize_unit_markdown_image_titles


def test_image_title_summary_counts_titles_duplicates_and_ignores_links_and_fences():
    summary = summarize_unit_markdown_image_titles(
        [
            {"id": "u1", "content": '![Alt](a.png "Hero") [Link](x "No")\n![No](b.png)'},
            {"id": "u2", "content": "```md\n![Code](c.png \"Hero\")\n```\n![Other](d.png 'hero')"},
        ]
    )

    assert summary["with_title"] == 2
    assert summary["without_title"] == 1
    assert summary["duplicate_titles"] == [{"title": "Hero", "image_count": 2}]
    assert [sample["alt_text"] for sample in summary["examples"]] == ["Alt", "No", "Other"]
