from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_attachment_gap_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_attachment_gap_csv_reports_expected_without_references():
    text = export_unit_attachment_gap_csv(
        [
            {"id": "missing", "source_project": "A", "metadata": {"attachment_count": 2}},
            {"id": "ok", "source_project": "A", "metadata": {"attachment_count": 1, "attachment_urls": ["https://example.test/a"]}},
        ]
    )

    result = rows(text)
    assert len(result) == 1
    assert result[0]["unit_id"] == "missing"
    assert result[0]["expected_attachment_count"] == "2"
    assert result[0]["found_attachment_count"] == "0"
    assert result[0]["gap_type"] == "missing_references"


def test_export_unit_attachment_gap_csv_reports_partial_references_and_path_mode(tmp_path):
    path = tmp_path / "gaps.csv"
    stats = export_unit_attachment_gap_csv(
        [{"id": "partial", "metadata": {"file_count": 2, "file_paths": ["/tmp/a.pdf"]}}],
        path,
    )

    row = rows(path.read_text(encoding="utf-8"))[0]
    assert row["source_project"] == "Unknown"
    assert row["gap_type"] == "partial_references"
    assert stats["rows_exported"] == 1

