from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_accessibility_metadata_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_accessibility_metadata_csv_reports_presence_score_and_missing_fields():
    text = export_unit_accessibility_metadata_csv(
        [
            {"id": "u2", "metadata": {"alt_text": "Chart", "transcript": "Words", "language": "en"}},
            {"id": "u1", "metadata": {"captions": "yes", "aria-label": "Play"}},
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "u1",
            "alt_text_present": "false",
            "transcript_present": "false",
            "captions_present": "true",
            "language_present": "false",
            "aria_label_present": "true",
            "accessibility_score": "0.40",
            "missing_fields": "alt_text; transcript; language",
        },
        {
            "unit_id": "u2",
            "alt_text_present": "true",
            "transcript_present": "true",
            "captions_present": "false",
            "language_present": "true",
            "aria_label_present": "false",
            "accessibility_score": "0.60",
            "missing_fields": "captions; aria_label",
        },
    ]


def test_export_unit_accessibility_metadata_csv_path_mode(tmp_path):
    path = tmp_path / "accessibility.csv"
    stats = export_unit_accessibility_metadata_csv([{"id": "u1"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["accessibility_score"] == "0.00"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
