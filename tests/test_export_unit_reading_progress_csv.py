from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_reading_progress_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_reading_progress_empty_input_has_header():
    assert export_unit_reading_progress_csv([]) == (
        "unit_id,title,status,pages_read,total_pages,progress_percent,started_at,completed_at\n"
    )


def test_unit_reading_progress_computes_percent_from_pages():
    result = rows(
        export_unit_reading_progress_csv(
            [{"id": "u1", "title": "Book", "metadata": {"pages_read": 50, "total_pages": 200}}]
        )
    )[0]

    assert result["pages_read"] == "50"
    assert result["total_pages"] == "200"
    assert result["progress_percent"] == "25"


def test_unit_reading_progress_uses_explicit_progress_and_completed_status():
    text = export_unit_reading_progress_csv(
        [
            {
                "id": "done",
                "title": "Done",
                "metadata": {"status": "completed", "completed_at": "2024-02-01T08:00:00Z"},
            },
            {
                "id": "explicit",
                "title": "Explicit",
                "metadata": {"progress": "0.375", "started_at": "2024-01-01"},
            },
        ]
    )

    result = {row["unit_id"]: row for row in rows(text)}
    assert result["done"]["progress_percent"] == "100"
    assert result["done"]["completed_at"] == "2024-02-01"
    assert result["explicit"]["progress_percent"] == "37.50"
    assert result["explicit"]["started_at"] == "2024-01-01"


def test_unit_reading_progress_handles_missing_and_nonnumeric_metadata(tmp_path):
    path = tmp_path / "progress.csv"
    stats = export_unit_reading_progress_csv(
        [{"id": "u1", "title": "Bad", "metadata": {"pages_read": "many", "total_pages": "none"}}],
        path,
    )

    result = rows(path.read_text(encoding="utf-8"))[0]
    assert result["pages_read"] == ""
    assert result["progress_percent"] == ""
    assert stats["rows_exported"] == 1
