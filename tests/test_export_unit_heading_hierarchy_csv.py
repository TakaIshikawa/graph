from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_heading_hierarchy_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_heading_hierarchy_csv_reports_parent_path_and_skips():
    rows = _rows(export_units_to_heading_hierarchy_csv([{"id": "u1", "title": "Note", "content": "# A\n### C\n## B\n```md\n# No\n```\n#### D"}]))

    assert rows == [
        {"unit_id": "u1", "title": "Note", "level": "1", "heading_text": "A", "line_number": "1", "parent_path": "", "skipped_level": "false"},
        {"unit_id": "u1", "title": "Note", "level": "3", "heading_text": "C", "line_number": "2", "parent_path": "A", "skipped_level": "true"},
        {"unit_id": "u1", "title": "Note", "level": "2", "heading_text": "B", "line_number": "3", "parent_path": "A", "skipped_level": "false"},
        {"unit_id": "u1", "title": "Note", "level": "4", "heading_text": "D", "line_number": "7", "parent_path": "A > B", "skipped_level": "true"},
    ]
