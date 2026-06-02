from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_image_title_attribute_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_image_title_attribute_csv_exports_all_quote_styles_and_ignores_untitled_images():
    text = export_units_to_markdown_image_title_attribute_csv(
        [
            {
                "id": "b",
                "title": "Beta",
                "source_project": "notes",
                "content": "![Plain](plain.png)\n![Single](single.png 'Single title')",
            },
            {
                "id": "a",
                "title": "Alpha",
                "source": "docs",
                "content": '![Double](https://img.test/a.png "Double title")\n![Paren](paren.png (Paren title))',
            },
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "source": "docs",
            "line_number": "1",
            "alt_text": "Double",
            "image_url": "https://img.test/a.png",
            "title_attribute": "Double title",
            "quote_style": "double",
        },
        {
            "unit_id": "a",
            "title": "Alpha",
            "source": "docs",
            "line_number": "2",
            "alt_text": "Paren",
            "image_url": "paren.png",
            "title_attribute": "Paren title",
            "quote_style": "parentheses",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "source": "notes",
            "line_number": "2",
            "alt_text": "Single",
            "image_url": "single.png",
            "title_attribute": "Single title",
            "quote_style": "single",
        },
    ]


def test_markdown_image_title_attribute_csv_preserves_escaped_characters():
    text = export_units_to_markdown_image_title_attribute_csv(
        [{"id": "u", "content": r'![A\]lt](image\ path.png "Title with \"quote\" and \) paren")'}]
    )

    assert _rows(text) == [
        {
            "unit_id": "u",
            "title": "",
            "source": "",
            "line_number": "1",
            "alt_text": "A]lt",
            "image_url": "image path.png",
            "title_attribute": 'Title with "quote" and ) paren',
            "quote_style": "double",
        }
    ]


def test_markdown_image_title_attribute_csv_ignores_code_spans_and_fences():
    content = "\n".join(
        [
            '`![Inline](inline.png "No")` ![Real](real.png "Yes")',
            "```",
            '![Fenced](fenced.png "No")',
            "```",
        ]
    )

    rows = _rows(export_units_to_markdown_image_title_attribute_csv([{"id": "u", "content": content}]))

    assert rows == [
        {
            "unit_id": "u",
            "title": "",
            "source": "",
            "line_number": "1",
            "alt_text": "Real",
            "image_url": "real.png",
            "title_attribute": "Yes",
            "quote_style": "double",
        }
    ]


def test_markdown_image_title_attribute_csv_path_mode_reports_write_metadata(tmp_path):
    path = tmp_path / "image-title-attributes.csv"
    units = [{"id": "u", "content": '![x](x.png "title")'}]

    expected = export_units_to_markdown_image_title_attribute_csv(units)
    stats = export_units_to_markdown_image_title_attribute_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
