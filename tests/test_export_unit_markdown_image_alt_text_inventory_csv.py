from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_image_alt_text_inventory_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_image_alt_text_inventory_exports_images_outside_fenced_code():
    text = export_units_to_markdown_image_alt_text_inventory_csv(
        [
            {
                "id": "b",
                "title": "Beta",
                "content": "![Logo](logo.png \"Brand mark\")\n```md\n![Ignored](ignored.png)\n```\n![](empty.png)",
            },
            {"id": "a", "title": "Alpha", "content": "Text\n![Chart](chart.svg 'Quarterly chart')"},
        ]
    )

    assert _rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "line_number": "2",
            "alt_text": "Chart",
            "destination": "chart.svg",
            "title_text": "Quarterly chart",
            "is_empty_alt": "false",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "line_number": "1",
            "alt_text": "Logo",
            "destination": "logo.png",
            "title_text": "Brand mark",
            "is_empty_alt": "false",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "line_number": "5",
            "alt_text": "",
            "destination": "empty.png",
            "title_text": "",
            "is_empty_alt": "true",
        },
    ]


def test_markdown_image_alt_text_inventory_path_mode_writes_stats(tmp_path):
    path = tmp_path / "images.csv"
    units = [{"unit_id": "u1", "metadata": {"title": "Meta"}, "content": "![Alt](a.png)"}]

    expected = export_units_to_markdown_image_alt_text_inventory_csv(units)
    stats = export_units_to_markdown_image_alt_text_inventory_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
