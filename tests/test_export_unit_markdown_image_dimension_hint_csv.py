import csv
from io import StringIO

from graph.export import export_units_to_markdown_image_dimension_hint_csv


def test_image_dimension_hint_export_detects_common_hint_styles():
    content = "\n".join(
        [
            "![a](photo.png =300x200)",
            "![b](https://example.com/chart.png|300)",
            "![c](img.png?width=640&height=480)",
        ]
    )

    rows = list(csv.DictReader(StringIO(export_units_to_markdown_image_dimension_hint_csv([{"id": "u1", "title": "Images", "content": content}]))))

    assert [(row["alt_text"], row["target"], row["width"], row["height"], row["hint_style"], row["line_number"]) for row in rows] == [
        ("a", "photo.png =300x200", "300", "200", "equals", "1"),
        ("b", "https://example.com/chart.png|300", "300", "", "pipe", "2"),
        ("c", "img.png?width=640&height=480", "640", "480", "attribute", "3"),
    ]


def test_image_dimension_hint_export_ignores_plain_images_and_fenced_code():
    content = "\n".join(
        [
            "![plain](photo.png)",
            "```",
            "![hidden](photo.png =10x20)",
            "```",
        ]
    )

    rows = list(csv.DictReader(StringIO(export_units_to_markdown_image_dimension_hint_csv([{"id": "u1", "content": content}]))))

    assert rows == []
