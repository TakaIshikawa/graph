from __future__ import annotations

import csv
from io import StringIO

from graph.export.unit_heading_duplicate_csv import export_units_to_heading_duplicate_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_heading_duplicate_csv_reports_duplicates_within_unit_only():
    text = export_units_to_heading_duplicate_csv(
        [
            {"id": "u1", "content": "# Intro!\n## Other\n# intro\n```md\n# Intro\n```\n### Other?"},
            {"id": "u2", "content": "# Intro"},
        ]
    )

    assert _rows(text) == [
        {"unit_id": "u1", "heading_text": "intro", "normalized_slug": "intro", "first_line": "1", "duplicate_line": "3", "level": "1"},
        {"unit_id": "u1", "heading_text": "Other?", "normalized_slug": "other", "first_line": "2", "duplicate_line": "7", "level": "3"},
    ]
