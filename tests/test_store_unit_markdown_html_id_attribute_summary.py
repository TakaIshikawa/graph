from graph.store.unit_markdown_html_id_attribute_summary import summarize_unit_markdown_html_id_attributes


def test_summarizes_html_ids_duplicates_and_ignores_comments_and_fences():
    result = summarize_unit_markdown_html_id_attributes(
        [
            {"id": "b", "content": "<div id=\"dup\"></div> <!-- <i id=\"skip\"></i> -->\n```\n<p id=\"skip\"></p>\n```"},
            {"id": "a", "content": "<span id='dup'></span> <section id=solo></section>"},
        ]
    )

    assert result == {
        "total_units": 2,
        "id_attribute_count": 3,
        "unique_id_count": 2,
        "duplicate_id_count": 1,
        "duplicate_ids": {"dup": 2},
        "samples": [
            {"unit_id": "a", "line_number": 1, "tag_name": "span", "id": "dup"},
            {"unit_id": "a", "line_number": 1, "tag_name": "section", "id": "solo"},
            {"unit_id": "b", "line_number": 1, "tag_name": "div", "id": "dup"},
        ],
        "units": [{"unit_id": "a", "id_attribute_count": 2}, {"unit_id": "b", "id_attribute_count": 1}],
    }
