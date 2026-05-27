from __future__ import annotations

from graph.store import summarize_unit_markdown_link_titles


def test_counts_titled_and_untitled_links_with_title_delimiters():
    report = summarize_unit_markdown_link_titles([{"id": "u", "content": '[A](https://a.test "Alpha") [B](https://b.test \'Beta\') [C](https://c.test (Gamma)) [D](https://d.test)'}])

    assert report["markdown_link_count"] == 4
    assert report["titled_link_count"] == 3
    assert report["untitled_link_count"] == 1
    assert [sample["link_title"] for sample in report["link_title_samples"][:3]] == ["Alpha", "Beta", "Gamma"]


def test_excludes_images_references_and_fenced_code_and_limits_samples():
    content = "![Alt](image.png)\n[Ref][id]\n```\n[Skip](https://skip.test \"s\")\n```\n[One](https://one.test \"1\")\n[Two](https://two.test \"2\")"

    report = summarize_unit_markdown_link_titles([{"id": "u", "content": content}], sample_limit=1)

    assert report["markdown_link_count"] == 2
    assert report["link_title_samples"] == [{"unit_id": "u", "line_number": 6, "text": "One", "url": "https://one.test", "link_title": "1"}]
