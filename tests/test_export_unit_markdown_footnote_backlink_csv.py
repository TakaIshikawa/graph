from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_footnote_backlinks_to_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_footnote_backlink_csv_reports_matched_missing_and_unused_definitions():
    text = export_unit_markdown_footnote_backlinks_to_csv(
        [{"id": "u", "title": "Unit", "content": "A[^one] B[^one] C[^missing]\n[^one]: ok\n[^unused]: no\n```\n[^skip]\n```"}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Unit", "footnote_label": "missing", "reference_count": "1", "definition_line": "", "first_reference_line": "1", "missing_definition": "true", "unused_definition": "false"},
        {"unit_id": "u", "title": "Unit", "footnote_label": "one", "reference_count": "2", "definition_line": "2", "first_reference_line": "1", "missing_definition": "false", "unused_definition": "false"},
        {"unit_id": "u", "title": "Unit", "footnote_label": "unused", "reference_count": "0", "definition_line": "3", "first_reference_line": "", "missing_definition": "false", "unused_definition": "true"},
    ]
