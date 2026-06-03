from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_duplicate_heading_anchor_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_duplicate_generated_heading_slugs_with_occurrence_indexes():
    result = rows(export_units_to_markdown_duplicate_heading_anchor_csv([{"id": "u", "content": "# Intro!\n## Other\n### intro"}]))

    assert [(row["slug"], row["heading_text"], row["line_number"], row["occurrence_index"], row["heading_level"]) for row in result] == [
        ("intro", "Intro!", "1", "1", "1"),
        ("intro", "intro", "3", "2", "3"),
    ]
