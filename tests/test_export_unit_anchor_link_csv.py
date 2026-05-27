from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_anchor_link_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_anchor_link_csv_matches_local_headings_and_flags_missing():
    rows = _rows(export_units_to_anchor_link_csv([{"id": "u", "content": "# Hello World\n[ok](#hello-world)\n[bad](note.md#missing)"}]))

    assert [(row["target_slug"], row["matched_heading"], row["unresolved"]) for row in rows] == [
        ("hello-world", "Hello World", "false"),
        ("missing", "", "true"),
    ]
