from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_empty_link_texts_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_empty_link_text_csv_exports_inline_and_reference_links_only():
    text = export_unit_markdown_empty_link_texts_to_csv(
        [
            {"id": "u", "title": "Unit", "content": "[](https://example.test) [ ][ref] [ok](x) ![](img.png)\n```\n[](ignored)\n```"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u", "title": "Unit", "line_number": "1", "link_type": "inline", "target": "https://example.test", "raw_link": "[](https://example.test)"},
        {"unit_id": "u", "title": "Unit", "line_number": "1", "link_type": "reference", "target": "ref", "raw_link": "[ ][ref]"},
    ]


def test_empty_link_text_csv_path_mode(tmp_path):
    path = tmp_path / "links.csv"

    stats = export_unit_markdown_empty_link_texts_to_csv([{"id": "u", "content": "[](x)"}], path)

    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
