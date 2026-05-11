from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.spotify_takeout import SpotifyTakeoutAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_spotify_takeout_ingests_standard_and_extended_history(tmp_path):
    standard = tmp_path / "StreamingHistory_music_0.json"
    standard.write_text(
        json.dumps(
            [
                {
                    "endTime": "2025-01-01 10:00",
                    "artistName": "Legacy Artist",
                    "trackName": "Legacy Track",
                    "msPlayed": 123456,
                }
            ]
        ),
        encoding="utf-8",
    )
    extended = tmp_path / "Streaming_History_Audio_2024-2025_0.json"
    extended.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-02T11:30:00Z",
                    "master_metadata_track_name": "Extended Track",
                    "master_metadata_album_artist_name": "Extended Artist",
                    "master_metadata_album_album_name": "Extended Album",
                    "spotify_track_uri": "spotify:track:abc123",
                    "ms_played": 654321,
                    "platform": "ios",
                    "conn_country": "US",
                    "reason_start": "trackdone",
                    "reason_end": "fwdbtn",
                    "shuffle": True,
                    "skipped": False,
                    "offline": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyTakeoutAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    legacy_unit = result.units[0]
    assert legacy_unit.source_project == SourceProject.SPOTIFY_TAKEOUT
    assert legacy_unit.source_entity_type == "play"
    assert legacy_unit.source_id.startswith("spotify_takeout:")
    assert legacy_unit.title == "Legacy Artist - Legacy Track"
    assert legacy_unit.content_type == ContentType.METADATA
    assert legacy_unit.tags == ["spotify", "music"]
    assert legacy_unit.created_at == datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert legacy_unit.metadata["played_at"] == "2025-01-01T10:00:00+00:00"
    assert legacy_unit.metadata["artist_name"] == "Legacy Artist"
    assert legacy_unit.metadata["track_name"] == "Legacy Track"
    assert legacy_unit.metadata["album_name"] == ""
    assert legacy_unit.metadata["ms_played"] == 123456
    assert legacy_unit.metadata["source_file"] == "StreamingHistory_music_0.json"
    assert "Played at: 2025-01-01T10:00:00+00:00" in legacy_unit.content
    assert "Track: Legacy Artist - Legacy Track" in legacy_unit.content

    extended_unit = result.units[1]
    assert extended_unit.title == "Extended Artist - Extended Track"
    assert extended_unit.metadata["artist_name"] == "Extended Artist"
    assert extended_unit.metadata["track_name"] == "Extended Track"
    assert extended_unit.metadata["album_name"] == "Extended Album"
    assert extended_unit.metadata["ms_played"] == 654321
    assert extended_unit.metadata["platform"] == "ios"
    assert extended_unit.metadata["country"] == "US"
    assert extended_unit.metadata["reason_start"] == "trackdone"
    assert extended_unit.metadata["reason_end"] == "fwdbtn"
    assert extended_unit.metadata["shuffle"] is True
    assert extended_unit.metadata["skipped"] is False
    assert extended_unit.metadata["offline"] is False
    assert extended_unit.metadata["spotify_uri"] == "spotify:track:abc123"


def test_spotify_takeout_reads_nested_extended_endsong_files_and_skips_other_json(tmp_path):
    nested = tmp_path / "Spotify Extended Streaming History"
    nested.mkdir()
    (tmp_path / "profile.json").write_text(json.dumps({"display_name": "Ada"}), encoding="utf-8")
    (nested / "endsong_0.json").write_text(
        json.dumps(
            [
                {
                    "ts": "2025-02-01T00:00:00Z",
                    "master_metadata_track_name": "Nested Track",
                    "master_metadata_album_artist_name": "Nested Artist",
                    "ms_played": 100,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyTakeoutAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].metadata["track_name"] == "Nested Track"
    assert result.units[0].metadata["source_file"] == "endsong_0.json"


def test_spotify_takeout_filters_by_sync_state(tmp_path):
    export = tmp_path / "StreamingHistory_music_0.json"
    export.write_text(
        json.dumps(
            [
                {"endTime": "2025-01-01 10:00", "artistName": "Old Artist", "trackName": "Old Track"},
                {"endTime": "2025-01-01 10:01", "artistName": "Boundary Artist", "trackName": "Boundary Track"},
                {"endTime": "2025-01-01 10:02", "artistName": "New Artist", "trackName": "New Track"},
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="spotify_takeout",
        source_entity_type="play",
        last_sync_at=datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc),
    )

    result = SpotifyTakeoutAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert result.units[0].metadata["track_name"] == "New Track"


def test_spotify_takeout_entity_type_filtering_does_not_load_plays(tmp_path):
    export = tmp_path / "StreamingHistory_music_0.json"
    export.write_text(
        json.dumps([{"endTime": "2025-01-01 10:00", "artistName": "Artist", "trackName": "Track"}]),
        encoding="utf-8",
    )

    result = SpotifyTakeoutAdapter(path=str(export)).ingest(entity_types=["artist"])

    assert result.units == []
    assert result.edges == []


def test_spotify_takeout_source_id_is_stable(tmp_path):
    export = tmp_path / "Streaming_History_Audio_2024-2025_0.json"
    event = {
        "ts": "2025-01-01T00:00:00Z",
        "master_metadata_track_name": "Stable Track",
        "master_metadata_album_artist_name": "Stable Artist",
        "spotify_track_uri": "spotify:track:stable",
        "ms_played": 42,
    }
    export.write_text(json.dumps([event]), encoding="utf-8")

    first = SpotifyTakeoutAdapter(path=str(export)).ingest().units[0]
    second = SpotifyTakeoutAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id


def test_spotify_takeout_adapter_is_registered():
    assert "spotify_takeout" in list_adapters()
    adapter = get_adapter("spotify-takeout", path="/tmp/spotify")
    assert isinstance(adapter, SpotifyTakeoutAdapter)
    assert adapter.name == "spotify_takeout"
