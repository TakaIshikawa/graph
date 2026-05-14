from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.netflix_viewing_activity_csv import NetflixViewingActivityCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows, encoding="utf-8"):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_netflix_viewing_activity_csv_ingests_rows(tmp_path):
    export = tmp_path / "ViewingActivity.csv"
    _write_csv(
        export,
        [
            {
                "Profile Name": "Taka",
                "Start Time": "2026-05-01 12:30:00",
                "Duration": "00:42:15",
                "Device Type": "Apple TV",
                "Country": "JP",
                "Title": "Example Show: Season 1: Pilot",
            }
        ],
    )

    result = NetflixViewingActivityCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.NETFLIX_VIEWING_ACTIVITY_CSV
    assert unit.source_entity_type == "watch_activity"
    assert unit.title == "Example Show: Season 1: Pilot"
    assert unit.created_at == datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)
    assert unit.metadata["profile_name"] == "Taka"
    assert unit.metadata["watched_at"] == "2026-05-01T12:30:00+00:00"
    assert unit.metadata["duration"] == "00:42:15"
    assert unit.metadata["duration_seconds"] == 2535
    assert unit.metadata["device"] == "Apple TV"
    assert unit.metadata["country"] == "JP"
    assert "Profile: Taka" in unit.content
    assert unit.source_id == NetflixViewingActivityCsvAdapter(path=str(export)).ingest().units[0].source_id


def test_netflix_viewing_activity_csv_directory_entity_filter_since_and_invalid_rows(tmp_path):
    _write_csv(
        tmp_path / "one.csv",
        [
            {"Profile Name": "A", "Date": "2026-04-30", "Title": "Old"},
            {"Profile Name": "A", "Date": "2026-05-02", "Title": "New"},
            {"Profile Name": "A", "Date": "2026-05-03", "Title": ""},
            {"Profile Name": "A", "Date": "not a date", "Title": "Bad Date"},
        ],
    )
    _write_csv(tmp_path / "two.csv", [{"Profile Name": "B", "Date": "2026-05-04", "Title": "Newest"}])
    since = SyncState(
        source_project="netflix_viewing_activity_csv",
        source_entity_type="watch_activity",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = NetflixViewingActivityCsvAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = NetflixViewingActivityCsvAdapter(path=str(tmp_path)).ingest(entity_types=["movie"])

    assert [unit.title for unit in result.units] == ["New", "Newest"]
    assert skipped.units == []
    assert get_adapter("netflix_viewing_activity_csv", path=str(tmp_path)).name == "netflix_viewing_activity_csv"


def test_netflix_viewing_activity_csv_reads_utf8_sig(tmp_path):
    export = tmp_path / "ViewingActivity.csv"
    _write_csv(export, [{"Profile Name": "A", "Date": "2026-05-01", "Title": "BOM Title"}], encoding="utf-8-sig")

    result = NetflixViewingActivityCsvAdapter(path=str(export)).ingest()

    assert [unit.title for unit in result.units] == ["BOM Title"]
