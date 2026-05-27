from __future__ import annotations

from graph.store import summarize_unit_markdown_image_alt_text


def test_summarize_unit_markdown_image_alt_text_empty_and_no_images():
    summary = summarize_unit_markdown_image_alt_text([{"id": "a", "content": ""}, {"id": "b", "content": "[link](x.png)"}])
    assert summary["image_count"] == 0
    assert summary["units"] == []


def test_summarize_unit_markdown_image_alt_text_counts_empty_and_present():
    summary = summarize_unit_markdown_image_alt_text([{"id": "a", "content": "![](a.png) ![Logo](b.png) ![  ](c.png)"}])
    assert summary["image_count"] == 3
    assert summary["empty_alt_count"] == 2
    assert summary["present_alt_count"] == 1
    assert summary["units"][0]["empty_alt_count"] == 2
