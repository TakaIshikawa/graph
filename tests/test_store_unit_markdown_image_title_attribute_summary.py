from graph.store.unit_markdown_image_title_attribute_summary import summarize_unit_markdown_image_title_attributes


def test_summarizes_image_title_coverage_and_excludes_regular_links():
    result = summarize_unit_markdown_image_title_attributes(
        [
            {"id": "b", "content": "[Link](x \"Title\") ![Alt](img.png \"Shown\")"},
            {"id": "a", "content": "![No](plain.png)\n![Yes](other.png 'Title')\n```\n![Skip](x.png)\n```"},
        ]
    )

    assert result == {
        "total_units": 2,
        "image_count": 3,
        "images_with_title_count": 2,
        "images_without_title_count": 1,
        "units_with_images": 2,
        "units_missing_titles": 1,
        "units": [
            {"unit_id": "a", "image_count": 2, "missing_title_count": 1},
            {"unit_id": "b", "image_count": 1, "missing_title_count": 0},
        ],
    }
