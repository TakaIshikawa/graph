from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_reading_velocity_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_reading_velocity_groups_progress_and_dates():
    text = export_collection_reading_velocity_csv(
        [
            {"id": "a", "metadata": {"collection": "Sci", "status": "completed", "updated_at": "2024-01-01"}},
            {"id": "b", "metadata": {"collection": "Sci", "pages_read": 50, "total_pages": 100, "updated_at": "2024-01-08"}},
            {"id": "c", "metadata": {"collections": ["Sci", "Ops"]}},
        ]
    )

    result = rows(text)
    assert result[0]["collection"] == "Ops"
    assert result[0]["total_units"] == "1"
    assert result[0]["average_progress_percent"] == "0.0"
    assert result[1]["collection"] == "Sci"
    assert result[1]["total_units"] == "3"
    assert result[1]["started_units"] == "2"
    assert result[1]["completed_units"] == "1"
    assert result[1]["average_progress_percent"] == "50.0"
    assert result[1]["latest_activity_date"] == "2024-01-08"


def test_collection_reading_velocity_handles_missing_collection_and_path_mode(tmp_path):
    path = tmp_path / "velocity.csv"
    stats = export_collection_reading_velocity_csv([{"id": "a", "metadata": {}}], path)

    assert stats["rows_exported"] == 1
    assert rows(path.read_text(encoding="utf-8"))[0] == {
        "collection": "unassigned",
        "total_units": "1",
        "started_units": "0",
        "completed_units": "0",
        "average_progress_percent": "0.0",
        "latest_activity_date": "",
        "units_per_week": "0.00",
    }
