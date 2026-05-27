from graph.store import summarize_unit_markdown_embeds


def test_summary_separates_obsidian_and_markdown_image_embeds():
    summary = summarize_unit_markdown_embeds([{"content": "![[img.PNG|Caption]]\n![](doc.pdf)\n![Alt](photo.jpg)"}])
    assert summary["total_embeds"] == 3
    assert summary["units_with_embeds"] == 1
    assert summary["syntax_counts"] == {"markdown_image": 2, "obsidian": 1}
    assert summary["extension_counts"] == {"jpg": 1, "pdf": 1, "png": 1}
    assert summary["missing_alt_or_caption_count"] == 1
