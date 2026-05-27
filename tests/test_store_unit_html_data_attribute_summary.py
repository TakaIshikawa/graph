from __future__ import annotations

from graph.store import summarize_unit_html_data_attributes


def test_counts_attributes_tags_and_values():
    report = summarize_unit_html_data_attributes([{"id": "u", "content": '<div data-a="1" data-b=\'2\'></div>\n<span data-a=3 data-empty></span>'}])

    assert report["html_data_attribute_count"] == 4
    assert report["attribute_counts"] == {"data-a": 2, "data-b": 1, "data-empty": 1}
    assert report["tag_counts"] == {"div": 2, "span": 2}


def test_fenced_code_and_sample_limit():
    content = "```\n<div data-skip=x></div>\n```\n<div data-one=1 data-two=\"\"></div>"

    report = summarize_unit_html_data_attributes([{"id": "u", "content": content}], sample_limit=1)

    assert report["html_data_attribute_count"] == 2
    assert report["data_attribute_samples"] == [{"unit_id": "u", "line_number": 4, "tag": "div", "attribute": "data-one", "value": "1"}]
