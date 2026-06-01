from graph.store.unit_markdown_fenced_div_summary import summarize_unit_markdown_fenced_divs


def test_summarizes_fenced_div_attributes_and_unmatched_closings():
    report = summarize_unit_markdown_fenced_divs([
        {"id": "b", "content": "::: {.note .wide #main key=value}\n:::\n:::\n```\n::: {.hidden}\n```"},
        {"id": "a", "content": "::: {.outer}\n::: {.inner role=doc}\n:::\n:::"},
    ])

    assert report["total_blocks"] == 3
    assert report["units_with_fenced_divs"] == 2
    assert report["class_counts"] == {"inner": 1, "note": 1, "outer": 1, "wide": 1}
    assert report["id_count"] == 1
    assert report["attribute_key_counts"] == {"key": 1, "role": 1}
    assert report["unmatched_closing_markers"] == 1
    assert report["samples"] == [
        {"unit_id": "a", "line_number": 1, "classes": ["outer"], "identifier": ""},
        {"unit_id": "a", "line_number": 2, "classes": ["inner"], "identifier": ""},
        {"unit_id": "b", "line_number": 1, "classes": ["note", "wide"], "identifier": "main"},
    ]
