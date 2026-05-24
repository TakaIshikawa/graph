from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.dropbox_activity_csv import DropboxActivityCsvAdapter
from graph.types.models import SyncState


def test_dropbox_activity_csv_ingests_common_headers(tmp_path):
    export = tmp_path / "dropbox.csv"
    export.write_text(
        "Date,Event,Path,Actor,Device,IP Address,Shared Folder\n"
        "2026-04-01 09:00:00,File added,/Team/Plan.pdf,ada@example.com,MacBook,192.0.2.4,Team\n",
        encoding="utf-8",
    )

    unit = DropboxActivityCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "dropbox_activity_csv"
    assert unit.source_entity_type == "activity"
    assert unit.metadata["event"] == "File added"
    assert unit.metadata["path"] == "/Team/Plan.pdf"
    assert unit.metadata["actor"] == "ada@example.com"
    assert unit.metadata["device"] == "MacBook"
    assert unit.metadata["ip_address"] == "192.0.2.4"
    assert unit.metadata["shared_folder"] == "Team"
    assert "dropbox" in unit.tags
    assert "Event: File added" in unit.content


def test_dropbox_activity_csv_alternate_headers_since_and_minimal_rows(tmp_path):
    export = tmp_path / "dropbox.csv"
    export.write_text(
        "Time,Action,File,User\n"
        "2026-04-03,shared,/new.txt,babbage@example.com\n"
        "2026-04-01,deleted,/old.txt,ada@example.com\n"
        ",,,\n"
        ",renamed,,\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="dropbox_activity_csv", source_entity_type="activity", last_sync_at=datetime(2026, 4, 2, tzinfo=timezone.utc))

    units = DropboxActivityCsvAdapter(path=str(export)).ingest(since=since).units

    assert [unit.metadata["event"] for unit in units] == ["shared"]
    all_units = DropboxActivityCsvAdapter(path=str(export)).ingest().units
    assert [unit.metadata["event"] for unit in all_units] == ["deleted", "shared", "renamed"]
    assert all_units[2].title == "renamed"
    assert DropboxActivityCsvAdapter(path=str(export)).ingest(entity_types=["note"]).units == []
