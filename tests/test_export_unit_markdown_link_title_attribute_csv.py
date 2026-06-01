from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_link_title_attributes_to_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_link_title_attribute_csv_detects_title_forms_and_skips_untitled_links():
    text = export_unit_markdown_link_title_attributes_to_csv(
        [
            {
                "id": "b",
                "title": "Beta",
                "content": "[double](https://b.test \"Title B\") [plain](https://skip.test)\n```\n[ignored](https://x.test 'No')\n```",
            },
            {
                "id": "a",
                "title": "Alpha",
                "content": "[single](https://a.test 'Title A')\n[parenthesized](https://p.test (  ))",
            },
        ]
    )

    assert _rows(text) == [
        {"unit_id": "a", "title": "Alpha", "link_text": "single", "target_url": "https://a.test", "title_text": "Title A", "line_number": "1", "empty_title": "false"},
        {"unit_id": "a", "title": "Alpha", "link_text": "parenthesized", "target_url": "https://p.test", "title_text": "", "line_number": "2", "empty_title": "true"},
        {"unit_id": "b", "title": "Beta", "link_text": "double", "target_url": "https://b.test", "title_text": "Title B", "line_number": "1", "empty_title": "false"},
    ]


def test_markdown_link_title_attribute_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "link_titles.csv"
    units = [{"id": "u", "content": "[x](https://example.test \"title\")"}]

    expected = export_unit_markdown_link_title_attributes_to_csv(units)
    stats = export_unit_markdown_link_title_attributes_to_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
