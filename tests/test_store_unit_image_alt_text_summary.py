from __future__ import annotations

from graph.store.unit_image_alt_text_summary import summarize_unit_image_alt_text


def test_image_alt_text_summary_distinguishes_images_from_links_and_blank_alt():
    report = summarize_unit_image_alt_text(
        [
            {"id": "a", "content": "![Diagram](a.png) and [not image](b.png)"},
            {"id": "b", "content": "![](empty.png) ![   ](blank.png)"},
        ]
    )

    assert report["total_units"] == 2
    assert report["total_images"] == 3
    assert report["present_alt_text_count"] == 1
    assert report["empty_alt_text_count"] == 2
    assert report["missing_alt_text_count"] == 0
    assert report["units_with_present_alt_text"] == ["a"]
    assert report["units_with_empty_alt_text"] == ["b"]
