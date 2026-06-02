from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_callout_titles_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_callout_title_csv_exports_titled_callouts_only():
    text = export_unit_markdown_callout_titles_to_csv(
        [
            {"id": "u", "title": "Unit", "content": "> [!note] Custom title\n> [!warning]+ Folded title\n> [!tip]\n```\n> [!note] Ignored\n```"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u", "title": "Unit", "line_number": "1", "callout_type": "note", "fold_marker": "", "callout_title": "Custom title"},
        {"unit_id": "u", "title": "Unit", "line_number": "2", "callout_type": "warning", "fold_marker": "+", "callout_title": "Folded title"},
    ]
