from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_kbd_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_kbd_csv_exports_entities_nested_tags_and_ignores_fences():
    content = "Use <kbd>Ctrl &amp; <span>K</span></kbd>\n```html\n<kbd>skip</kbd>\n```\n<kbd>Esc</kbd>"

    result = rows(export_unit_markdown_html_kbd_csv([{"id": "u", "title": "T", "source": "s", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "text": "Ctrl & K", "nested_tag_count": "2"},
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "5", "text": "Esc", "nested_tag_count": "0"},
    ]


def test_kbd_csv_path_write_returns_metadata(tmp_path):
    output = tmp_path / "kbd.csv"

    result = export_unit_markdown_html_kbd_csv([{"id": "u", "content": "<kbd>K</kbd>"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
