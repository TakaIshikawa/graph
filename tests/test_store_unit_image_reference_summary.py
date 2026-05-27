from __future__ import annotations

from graph.store.unit_image_reference_summary import summarize_unit_image_references


def test_image_reference_summary_inventories_markdown_and_html_images():
    report = summarize_unit_image_references(
        [
            {"content": "![Logo](assets/logo.PNG) [link](x.png)"},
            {"content": '<img src="https://example.com/a.jpg" alt=""> <img src="/b.svg">'},
        ]
    )

    assert report["total_images"] == 3
    assert report["markdown_image_count"] == 1
    assert report["html_image_count"] == 2
    assert report["remote_image_count"] == 1
    assert report["local_image_count"] == 2
    assert report["populated_alt_text_count"] == 1
    assert report["blank_alt_text_count"] == 1
    assert report["missing_alt_text_count"] == 1
    assert report["scheme_counts"] == [{"scheme": "https", "count": 1}, {"scheme": "local", "count": 2}]
    assert report["extension_counts"] == [
        {"extension": "jpg", "count": 1},
        {"extension": "png", "count": 1},
        {"extension": "svg", "count": 1},
    ]
