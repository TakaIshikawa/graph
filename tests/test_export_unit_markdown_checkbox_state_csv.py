from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_checkbox_state_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_checkbox_state_csv_classifies_supported_markers_and_ignores_fences():
    text = export_units_to_markdown_checkbox_state_csv(
        [
            {
                "id": "a",
                "title": "Alpha",
                "content": "- [ ] open\n- [x] done\n1. [-] blocked\n* [?] unknown\n```\n- [ ] ignored\n```",
            }
        ]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "line_number": "1", "marker": "-", "state": "open", "text": "open"},
        {"unit_id": "a", "title": "Alpha", "line_number": "2", "marker": "-", "state": "done", "text": "done"},
        {"unit_id": "a", "title": "Alpha", "line_number": "3", "marker": "1.", "state": "blocked", "text": "blocked"},
        {"unit_id": "a", "title": "Alpha", "line_number": "4", "marker": "*", "state": "unknown", "text": "unknown"},
    ]


def test_markdown_checkbox_state_csv_path_mode(tmp_path):
    path = tmp_path / "tasks.csv"
    stats = export_units_to_markdown_checkbox_state_csv([{"id": "a", "content": "- [X] done"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["state"] == "done"
    assert stats["rows_exported"] == 1
