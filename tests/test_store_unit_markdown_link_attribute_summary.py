from __future__ import annotations

from graph.store.unit_markdown_link_attribute_summary import summarize_unit_markdown_link_attributes


def test_detects_markdown_links_with_attribute_blocks():
    summary = summarize_unit_markdown_link_attributes(
        [
            {
                "id": "u1",
                "content": '[Docs](https://example.com){target=_blank rel=noreferrer}\n```\n[No](x){rel=noopener}\n```',
            }
        ]
    )

    assert summary["link_attribute_count"] == 1
    assert summary["units_with_link_attributes"] == 1
    assert summary["attribute_counts"] == {"rel": 1, "target": 1}
    assert summary["samples"] == [
        {"unit_id": "u1", "line_number": 1, "url": "https://example.com", "attributes": ["rel", "target"]}
    ]
