from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.spotify_streaming_history import SpotifyStreamingHistoryAdapter
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import SyncState


def test_spotify_streaming_history_ingests_legacy_and_extended_formats(tmp_path):
    legacy = tmp_path / "StreamingHistory_music_0.json"
    legacy.write_text(
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
    extended = tmp_path / "endsong_0.json"
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

    result = SpotifyStreamingHistoryAdapter(path=str(tmp_path)).ingest()

    plays = [unit for unit in result.units if unit.source_entity_type == "play"]
    assert len(plays) == 2
    legacy_unit = plays[0]
    assert legacy_unit.source_project == SourceProject.SPOTIFY_STREAMING_HISTORY
    assert legacy_unit.source_entity_type == "play"
    assert legacy_unit.source_id.startswith("spotify_streaming_history:")
    assert legacy_unit.title == "Legacy Track - Legacy Artist"
    assert legacy_unit.content_type == ContentType.METADATA
    assert legacy_unit.tags == ["spotify", "music", "listening"]
    assert legacy_unit.created_at == datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert legacy_unit.metadata["track_name"] == "Legacy Track"
    assert legacy_unit.metadata["artist_name"] == "Legacy Artist"
    assert legacy_unit.metadata["album_name"] == ""
    assert legacy_unit.metadata["spotify_uri"] == ""
    assert legacy_unit.metadata["ms_played"] == 123456
    assert legacy_unit.metadata["source_file"] == "StreamingHistory_music_0.json"

    extended_unit = plays[1]
    assert extended_unit.title == "Extended Track - Extended Artist"
    assert extended_unit.metadata["track_name"] == "Extended Track"
    assert extended_unit.metadata["artist_name"] == "Extended Artist"
    assert extended_unit.metadata["album_name"] == "Extended Album"
    assert extended_unit.metadata["spotify_uri"] == "spotify:track:abc123"
    assert extended_unit.metadata["ms_played"] == 654321
    assert extended_unit.metadata["platform"] == "ios"
    assert extended_unit.metadata["country"] == "US"
    assert extended_unit.metadata["reason_start"] == "trackdone"
    assert extended_unit.metadata["reason_end"] == "fwdbtn"
    assert extended_unit.metadata["shuffle"] is True
    assert extended_unit.metadata["skipped"] is False
    assert extended_unit.metadata["offline"] is False
    assert extended_unit.metadata["source_file"] == "endsong_0.json"
    assert len([unit for unit in result.units if unit.source_entity_type == "session"]) == 2
    assert len(result.edges) == 2


def test_spotify_streaming_history_reads_matching_files_in_name_order(tmp_path):
    (tmp_path / "notes.json").write_text("[]", encoding="utf-8")
    (tmp_path / "StreamingHistory_music_2.json").write_text(
        json.dumps(
            [
                {
                    "endTime": "2025-01-03 00:00",
                    "artistName": "Artist C",
                    "trackName": "Track C",
                    "msPlayed": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "StreamingHistory_music_1.json").write_text(
        json.dumps(
            [
                {
                    "endTime": "2025-01-01 00:00",
                    "artistName": "Artist A",
                    "trackName": "Track A",
                    "msPlayed": 1,
                },
                {
                    "endTime": "2025-01-02 00:00",
                    "artistName": "Artist B",
                    "trackName": "Track B",
                    "msPlayed": 2,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyStreamingHistoryAdapter(path=str(tmp_path)).ingest()

    plays = [unit for unit in result.units if unit.source_entity_type == "play"]
    assert [unit.metadata["source_file"] for unit in plays] == [
        "StreamingHistory_music_1.json",
        "StreamingHistory_music_1.json",
        "StreamingHistory_music_2.json",
    ]
    assert [unit.metadata["track_name"] for unit in plays] == [
        "Track A",
        "Track B",
        "Track C",
    ]


def test_spotify_streaming_history_skips_malformed_files(tmp_path):
    bad = tmp_path / "StreamingHistory_music_bad.json"
    bad.write_text("not json", encoding="utf-8")
    good = tmp_path / "endsong_good.json"
    good.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-02-01T00:00:00Z",
                    "master_metadata_track_name": "Good Track",
                    "master_metadata_album_artist_name": "Good Artist",
                    "ms_played": 100,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyStreamingHistoryAdapter(path=str(tmp_path)).ingest()

    plays = [unit for unit in result.units if unit.source_entity_type == "play"]
    assert len(plays) == 1
    assert plays[0].metadata["track_name"] == "Good Track"


def test_spotify_streaming_history_filters_by_sync_state(tmp_path):
    export = tmp_path / "StreamingHistory_music_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "endTime": "2025-01-01 10:00",
                    "artistName": "Old Artist",
                    "trackName": "Old Track",
                    "msPlayed": 10,
                },
                {
                    "endTime": "2025-01-01 10:01",
                    "artistName": "Boundary Artist",
                    "trackName": "Boundary Track",
                    "msPlayed": 20,
                },
                {
                    "endTime": "2025-01-01 10:02",
                    "artistName": "New Artist",
                    "trackName": "New Track",
                    "msPlayed": 30,
                },
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="spotify_streaming_history",
        source_entity_type="play",
        last_sync_at=datetime(2025, 1, 1, 10, 1, tzinfo=timezone.utc),
    )

    result = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(since=since, entity_types=["play"])

    assert len(result.units) == 1
    assert result.units[0].metadata["track_name"] == "New Track"


def test_spotify_streaming_history_entity_type_filtering(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T00:00:00Z",
                    "master_metadata_track_name": "Filtered Track",
                    "master_metadata_album_artist_name": "Filtered Artist",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["album"])

    assert result.units == []
    assert result.edges == []


def test_spotify_streaming_history_source_id_is_stable(tmp_path):
    export = tmp_path / "endsong_0.json"
    event = {
        "ts": "2025-01-01T00:00:00Z",
        "master_metadata_track_name": "Stable Track",
        "master_metadata_album_artist_name": "Stable Artist",
        "ms_played": 42,
    }
    export.write_text(json.dumps([event]), encoding="utf-8")

    first = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["play"]).units[0]
    second = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["play"]).units[0]

    assert first.source_id == second.source_id


def test_spotify_streaming_history_sessions_and_filters(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T10:00:00Z",
                    "master_metadata_track_name": "One",
                    "master_metadata_album_artist_name": "Artist A",
                    "ms_played": 100,
                },
                {
                    "ts": "2025-01-01T10:20:00Z",
                    "master_metadata_track_name": "Two",
                    "master_metadata_album_artist_name": "Artist B",
                    "ms_played": 200,
                },
                {
                    "ts": "2025-01-01T11:00:01Z",
                    "master_metadata_track_name": "Three",
                    "master_metadata_album_artist_name": "Artist A",
                    "ms_played": 300,
                },
            ]
        ),
        encoding="utf-8",
    )

    combined = SpotifyStreamingHistoryAdapter(path=str(export)).ingest()
    sessions = [unit for unit in combined.units if unit.source_entity_type == "session"]
    assert [unit.metadata["play_count"] for unit in sessions] == [2, 1]
    assert sessions[0].metadata["total_ms_played"] == 300
    assert sessions[0].metadata["artist_count"] == 2
    assert sessions[0].metadata["track_count"] == 2
    assert len(combined.edges) == 3
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in combined.edges)

    session_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["session"])
    play_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["play"])
    assert {unit.source_entity_type for unit in session_only.units} == {"session"}
    assert session_only.edges == []
    assert {unit.source_entity_type for unit in play_only.units} == {"play"}
    assert play_only.edges == []


def test_spotify_streaming_history_adapter_is_registered():
    assert "spotify_streaming_history" in list_adapters()
    adapter = get_adapter("spotify_streaming_history", path="/tmp/spotify")
    assert adapter.name == "spotify_streaming_history"
