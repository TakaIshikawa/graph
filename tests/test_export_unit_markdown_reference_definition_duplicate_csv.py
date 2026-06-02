from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_reference_definition_duplicates_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_reference_definition_duplicate_csv_reports_each_duplicate_with_first_line():
    text = export_unit_markdown_reference_definition_duplicates_to_csv(
        [
            {"id": "u", "title": "Unit", "content": "[Ref  One]: https://first.test\n[other]: https://other.test\n[ref one]: https://second.test\n```\n[REF ONE]: https://ignored.test\n```"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u", "title": "Unit", "label": "Ref One", "normalized_label": "ref one", "first_line_number": "1", "duplicate_line_number": "3", "duplicate_target": "https://second.test"},
    ]
