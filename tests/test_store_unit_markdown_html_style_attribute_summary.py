from graph.store.unit_markdown_html_style_attribute_summary import summarize_unit_markdown_html_style_attributes


def test_summarizes_style_attributes_properties_and_ignores_comments_and_fences():
    result = summarize_unit_markdown_html_style_attributes(
        [
            {"id": "u", "content": "<span style=\"Color: red; font-weight: bold\"></span>\n<p style='color: blue'></p>\n<!-- <i style=\"bad: yes\"></i> -->\n```\n<b style=\"skip: yes\"></b>\n```"}
        ]
    )

    assert result["total_units"] == 1
    assert result["units_with_style_attributes"] == 1
    assert result["style_attribute_count"] == 2
    assert result["property_count"] == 3
    assert result["top_properties"][:2] == [{"property": "color", "count": 2}, {"property": "font-weight", "count": 1}]
    assert result["units"] == [{"unit_id": "u", "style_attribute_count": 2, "property_count": 3}]
