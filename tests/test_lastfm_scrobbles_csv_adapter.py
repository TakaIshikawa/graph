from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.lastfm_scrobbles_csv import LastfmScrobblesCsvAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_lastfm_scrobbles_csv_ingests_common_columns(tmp_path):
    export = tmp_path / "scrobbles.csv"
    _write_csv(
        export,
        [
            {
                "artist": "Björk",
                "album": "Homogenic",
                "track": "Jóga",
                "timestamp": "2025-01-02T03:04:05Z",
                "duration": "5:05",
                "track_url": "https://www.last.fm/music/Bjork/_/Joga",
                "album_artist": "Björk",
                "loved": "1",
                "skipped": "false",
            }
        ],
    )

    result = LastfmScrobblesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.LASTFM_SCROBBLES_CSV
    assert unit.source_entity_type == "play"
    assert unit.source_id.startswith("lastfm_scrobbles_csv:play:")
    assert unit.title == "Jóga - Björk"
    assert unit.content_type == ContentType.METADATA
    assert unit.tags == ["lastfm", "music", "listening"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.metadata["artist"] == "Björk"
    assert unit.metadata["album"] == "Homogenic"
    assert unit.metadata["track"] == "Jóga"
    assert unit.metadata["listened_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["duration"] == "5:05"
    assert unit.metadata["duration_seconds"] == 305
    assert unit.metadata["track_url"].endswith("/Joga")
    assert unit.metadata["album_artist"] == "Björk"
    assert unit.metadata["loved"] is True
    assert unit.metadata["skipped"] is False
    assert unit.metadata["source_file"] == "scrobbles.csv"
    assert unit.metadata["row"]["duration"] == "5:05"


def test_lastfm_scrobbles_csv_reads_directory_and_alias_columns(tmp_path):
    _write_csv(
        tmp_path / "one.csv",
        [
            {
                "Artist Name": "Nina Simone",
                "Album Name": "Pastel Blues",
                "Title": "Sinnerman",
                "Date": "2025-01-01 00:00:00",
                "URL": "https://www.last.fm/music/Nina+Simone/_/Sinnerman",
            }
        ],
    )
    _write_csv(
        tmp_path / "two.csv",
        [
            {
                "artist": "Kate Bush",
                "album": "Hounds of Love",
                "name": "Running Up That Hill",
                "uts": "1735776000",
                "duration_ms": "300000",
            }
        ],
    )

    result = LastfmScrobblesCsvAdapter(path=str(tmp_path)).ingest(entity_types=["play"])

    assert [unit.metadata["source_file"] for unit in result.units] == ["one.csv", "two.csv"]
    assert result.units[0].metadata["track"] == "Sinnerman"
    assert result.units[1].metadata["duration_seconds"] == 300


def test_lastfm_scrobbles_csv_filters_since_at_or_before_last_sync(tmp_path):
    export = tmp_path / "scrobbles.csv"
    _write_csv(
        export,
        [
            {"artist": "Old", "track": "Old Track", "timestamp": "2025-01-01T10:00:00Z"},
            {"artist": "Boundary", "track": "Boundary Track", "timestamp": "2025-01-01T10:01:00Z"},
            {"artist": "New", "track": "New Track", "timestamp": "2025-01-01T10:02:00Z"},
        ],
    )
    since = SyncState(
        source_project="lastfm_scrobbles_csv",
        source_entity_type="play",
        last_sync_at=datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc),
    )

    result = LastfmScrobblesCsvAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert result.units[0].metadata["track"] == "New Track"


def test_lastfm_scrobbles_csv_adds_repeat_sequence_metadata(tmp_path):
    export = tmp_path / "scrobbles.csv"
    _write_csv(
        export,
        [
            {"timestamp": "2025-03-01T00:00:00Z", "artist": "Björk", "album": "Homogenic", "track": "Jóga"},
            {"timestamp": "2025-01-01T00:00:00Z", "artist": "Björk", "album": "Homogenic", "track": "Jóga"},
            {"timestamp": "2025-02-01T00:00:00Z", "artist": "Björk", "album": "Homogenic", "track": "Jóga"},
            {"timestamp": "2025-02-02T00:00:00Z", "artist": "Björk", "album": "Debut", "track": "Human Behaviour"},
        ],
    )

    units = LastfmScrobblesCsvAdapter(path=str(export)).ingest().units

    assert [unit.metadata["play_sequence"] for unit in units] == [1, 2, 1, 3]
    assert [unit.metadata["is_replay"] for unit in units] == [False, True, False, True]
    assert "previous_played_at" not in units[0].metadata
    assert units[1].metadata["previous_played_at"] == "2025-01-01T00:00:00+00:00"
    assert units[3].metadata["previous_played_at"] == "2025-02-01T00:00:00+00:00"


def test_lastfm_scrobbles_csv_entity_type_filtering(tmp_path):
    export = tmp_path / "scrobbles.csv"
    _write_csv(export, [{"timestamp": "2025-01-01T00:00:00Z", "artist": "A", "track": "T"}])

    assert LastfmScrobblesCsvAdapter(path=str(export)).ingest(entity_types=["track"]).units == []
