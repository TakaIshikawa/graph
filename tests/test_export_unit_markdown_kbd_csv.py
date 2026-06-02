from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_markdown_kbd_csv import export_unit_markdown_kbd_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_repeated_keys_case_insensitive_closing_and_skips_fences():
    text = export_unit_markdown_kbd_csv(
        [
            {
                "id": "b",
                "title": "Beta",
                "content": "Use <kbd>Ctrl</KBD> and <kbd>Ctrl</kbd>\n```\n<kbd>Skip</kbd>\n```\nThen <kbd>Alt + F</kbd>",
            },
            {"id": "a", "title": "Alpha", "content": "Press <kbd>Esc</kbd>"},
        ]
    )

    assert _rows(text) == [
        {"unit_id": "a", "title": "Alpha", "line_number": "1", "key_text": "Esc", "raw_html": "<kbd>Esc</kbd>"},
        {"unit_id": "b", "title": "Beta", "line_number": "1", "key_text": "Ctrl", "raw_html": "<kbd>Ctrl</KBD>"},
        {"unit_id": "b", "title": "Beta", "line_number": "1", "key_text": "Ctrl", "raw_html": "<kbd>Ctrl</kbd>"},
        {"unit_id": "b", "title": "Beta", "line_number": "5", "key_text": "Alt + F", "raw_html": "<kbd>Alt + F</kbd>"},
    ]


def test_writes_csv_and_returns_stats(tmp_path):
    output = tmp_path / "kbd.csv"
    result = export_unit_markdown_kbd_csv([{"id": "u", "title": "Unit", "content": "<kbd>K</kbd>"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
    assert _rows(output.read_text(encoding="utf-8"))[0]["key_text"] == "K"
