from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_units_to_timestamp_completeness_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_timestamp_completeness_handles_datetime_strings_and_invalids():
    result = rows(export_units_to_timestamp_completeness_csv([
        {
            "id": "u",
            "created_at": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "updated_at": "2024-01-03T00:00:00Z",
            "metadata": {"published_at": "bad"},
        }
    ]))[0]

    assert result["earliest_timestamp"].startswith("2024-01-01T00:00:00")
    assert result["latest_timestamp"].startswith("2024-01-03T00:00:00")
    assert result["completeness_score"] == "0.50"
    assert result["invalid_timestamp_fields"] == "published_at"
