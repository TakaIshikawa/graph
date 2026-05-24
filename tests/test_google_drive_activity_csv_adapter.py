from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.google_drive_activity_csv import GoogleDriveActivityCsvAdapter
from graph.types.models import SyncState


def test_google_drive_activity_csv_ingests_common_headers(tmp_path):
    export = tmp_path / "drive.csv"
    export.write_text(
        "Time,File Name,Actor,Action,Owner,MIME Type,URL,File ID\n"
        "2026-04-01T10:00:00Z,Roadmap,ada@example.com,edited,owner@example.com,application/vnd.google-apps.document,https://docs.example/doc,doc-1\n",
        encoding="utf-8",
    )

    unit = GoogleDriveActivityCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "google_drive_activity_csv"
    assert unit.source_entity_type == "activity"
    assert unit.metadata["file_name"] == "Roadmap"
    assert unit.metadata["actor"] == "ada@example.com"
    assert unit.metadata["action"] == "edited"
    assert unit.metadata["mime_type"] == "application/vnd.google-apps.document"
    assert unit.metadata["url"] == "https://docs.example/doc"
    assert unit.tags == ["google-drive", "activity", "edited", "ada@example.com", "application/vnd.google-apps.document"]
    assert "Action: edited" in unit.content


def test_google_drive_activity_csv_alternate_headers_since_order_and_minimal_rows(tmp_path):
    export = tmp_path / "drive.csv"
    export.write_text(
        "Date,Title,User,Event\n"
        "2026-04-03,Newer,babbage@example.com,viewed\n"
        "2026-04-01,Older,ada@example.com,created\n"
        ",,,\n"
        ",Only title,,\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="google_drive_activity_csv", source_entity_type="activity", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    units = GoogleDriveActivityCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.metadata["file_name"] for unit in units] == ["Newer"]
    assert units[0].metadata["actor"] == "babbage@example.com"
    assert GoogleDriveActivityCsvAdapter(path=str(export)).ingest(entity_types=["note"]).units == []
    all_units = GoogleDriveActivityCsvAdapter(path=str(export)).ingest().units
    assert [unit.metadata["file_name"] for unit in all_units] == ["Older", "Newer", "Only title"]
    assert all_units[2].title == "Google Drive activity: Only title"
