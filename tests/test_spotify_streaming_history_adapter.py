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
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "session_contains_play"]) == 2


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

    result = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["genre"])

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
    assert len([edge for edge in combined.edges if edge.metadata["relation_type"] == "session_contains_play"]) == 3
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in combined.edges)

    session_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["session"])
    play_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["play"])
    assert {unit.source_entity_type for unit in session_only.units} == {"session"}
    assert session_only.edges == []
    assert {unit.source_entity_type for unit in play_only.units} == {"play"}
    assert play_only.edges == []


def test_spotify_streaming_history_listening_day_aggregates_and_filters(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T23:50:00-01:00",
                    "master_metadata_track_name": "Late Local",
                    "master_metadata_album_artist_name": "Artist A",
                    "ms_played": 100,
                },
                {
                    "ts": "2025-01-02T01:00:00Z",
                    "master_metadata_track_name": "UTC Track",
                    "master_metadata_album_artist_name": "Artist A",
                    "ms_played": 200,
                },
                {
                    "ts": "2025-01-02T02:00:00Z",
                    "episode_name": "Episode One",
                    "episode_show_name": "Show A",
                    "spotify_episode_uri": "spotify:episode:one",
                    "ms_played": 300,
                },
            ]
        ),
        encoding="utf-8",
    )

    combined = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["listening_day", "play"])
    days = [unit for unit in combined.units if unit.source_entity_type == "listening_day"]
    plays = [unit for unit in combined.units if unit.source_entity_type == "play"]

    assert len(days) == 2
    music_day = next(unit for unit in days if unit.metadata["media_kind"] == "music")
    podcast_day = next(unit for unit in days if unit.metadata["media_kind"] == "podcast")
    assert music_day.source_id.startswith("spotify_streaming_history:listening_day:")
    assert music_day.metadata["date"] == "2025-01-02"
    assert music_day.metadata["play_count"] == 2
    assert music_day.metadata["total_ms_played"] == 300
    assert music_day.metadata["unique_track_count"] == 2
    assert music_day.metadata["unique_artist_count"] == 1
    assert music_day.metadata["unique_episode_count"] == 0
    assert music_day.metadata["source_files"] == ["endsong_0.json"]
    assert podcast_day.metadata["unique_episode_count"] == 1
    assert len([edge for edge in combined.edges if edge.metadata["relation_type"] == "listening_day_contains_play"]) == len(plays)

    day_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["listening_day"])
    play_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["play"])
    assert {unit.source_entity_type for unit in day_only.units} == {"listening_day"}
    assert day_only.edges == []
    assert {unit.source_entity_type for unit in play_only.units} == {"play"}
    assert play_only.edges == []


def test_spotify_streaming_history_artist_and_track_aggregates(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T10:00:00Z",
                    "master_metadata_track_name": "Repeat",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Album 1",
                    "spotify_track_uri": "spotify:track:repeat",
                    "ms_played": 100,
                    "conn_country": "US",
                },
                {
                    "ts": "2025-01-02T10:00:00Z",
                    "master_metadata_track_name": "Repeat",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Album 1",
                    "spotify_track_uri": "spotify:track:repeat",
                    "ms_played": 200,
                    "conn_country": "JP",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["artist", "track", "play"])

    artist = next(unit for unit in result.units if unit.source_entity_type == "artist")
    track = next(unit for unit in result.units if unit.source_entity_type == "track")
    assert artist.metadata["artist_name"] == "Artist A"
    assert artist.metadata["play_count"] == 2
    assert artist.metadata["total_ms_played"] == 300
    assert artist.metadata["first_played_at"] == "2025-01-01T10:00:00+00:00"
    assert artist.metadata["last_played_at"] == "2025-01-02T10:00:00+00:00"
    assert artist.metadata["countries"] == ["JP", "US"]
    assert track.metadata["track_name"] == "Repeat"
    assert track.metadata["play_count"] == 2
    assert track.metadata["albums"] == ["Album 1"]
    assert {edge.metadata["relation_type"] for edge in result.edges} == {
        "artist_contains_track",
        "track_contains_play",
    }
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "track_contains_play"]) == 2


def test_spotify_streaming_history_album_aggregates_and_edges(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T10:00:00Z",
                    "master_metadata_track_name": "One",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Album 1",
                    "spotify_track_uri": "spotify:track:one",
                    "spotify_album_uri": "spotify:album:album1",
                    "ms_played": 100,
                },
                {
                    "ts": "2025-01-02T10:00:00Z",
                    "master_metadata_track_name": "Two",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Album 1",
                    "spotify_track_uri": "spotify:track:two",
                    "spotify_album_uri": "spotify:album:album1",
                    "ms_played": 200,
                },
                {
                    "ts": "2025-01-03T10:00:00Z",
                    "master_metadata_track_name": "One",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Album 1",
                    "spotify_track_uri": "spotify:track:one",
                    "spotify_album_uri": "spotify:album:album1",
                    "ms_played": 300,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["album", "track", "play"])

    albums = [unit for unit in result.units if unit.source_entity_type == "album"]
    assert len(albums) == 1
    album = albums[0]
    assert album.title == "Album 1 - Artist A"
    assert album.source_id.startswith("spotify_streaming_history:album:")
    assert album.metadata["spotify_album_uri"] == "spotify:album:album1"
    assert album.metadata["play_count"] == 3
    assert album.metadata["track_names"] == ["One", "Two"]
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "album_contains_track"]) == 2
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "album_contains_play"]) == 3


def test_spotify_streaming_history_album_fallback_identity_uses_album_and_artist(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T10:00:00Z",
                    "master_metadata_track_name": "One",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "Shared Name",
                },
                {
                    "ts": "2025-01-02T10:00:00Z",
                    "master_metadata_track_name": "Two",
                    "master_metadata_album_artist_name": "Artist A",
                    "master_metadata_album_album_name": "shared name",
                },
                {
                    "ts": "2025-01-03T10:00:00Z",
                    "master_metadata_track_name": "Other",
                    "master_metadata_album_artist_name": "Artist B",
                    "master_metadata_album_album_name": "Shared Name",
                },
            ]
        ),
        encoding="utf-8",
    )

    albums = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["album"]).units

    assert len(albums) == 2
    assert sorted(unit.metadata["play_count"] for unit in albums) == [1, 2]


def test_spotify_streaming_history_ingests_podcast_show_and_episode(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T10:00:00Z",
                    "episode_name": "Episode One",
                    "episode_show_name": "A Good Show",
                    "spotify_episode_uri": "spotify:episode:one",
                    "spotify_show_uri": "spotify:show:good",
                    "ms_played": 1000,
                },
                {
                    "ts": "2025-01-02T10:00:00Z",
                    "master_metadata_episode_name": "Episode One",
                    "master_metadata_show_name": "A Good Show",
                    "spotify_episode_uri": "spotify:episode:one",
                    "spotify_show_uri": "spotify:show:good",
                    "ms_played": 2000,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(
        entity_types=["play", "podcast_show", "podcast_episode"]
    )

    plays = [unit for unit in result.units if unit.source_entity_type == "play"]
    shows = [unit for unit in result.units if unit.source_entity_type == "podcast_show"]
    episodes = [unit for unit in result.units if unit.source_entity_type == "podcast_episode"]
    assert len(plays) == 2
    assert len(shows) == 1
    assert len(episodes) == 1
    assert plays[0].title == "Episode One - A Good Show"
    assert plays[0].metadata["media_kind"] == "podcast"
    assert shows[0].metadata["show_name"] == "A Good Show"
    assert shows[0].metadata["play_count"] == 2
    assert episodes[0].metadata["episode_name"] == "Episode One"
    assert episodes[0].metadata["total_ms_played"] == 3000
    assert {edge.metadata["relation_type"] for edge in result.edges} == {
        "podcast_show_contains_episode",
        "podcast_episode_contains_play",
        "podcast_show_contains_play",
    }
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "podcast_episode_contains_play"]) == 2


def test_spotify_streaming_history_podcast_edges_follow_entity_filters(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T10:00:00Z",
                    "episode_name": "Episode",
                    "episode_show_name": "Show",
                }
            ]
        ),
        encoding="utf-8",
    )

    episode_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["podcast_episode"])
    play_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["play"])

    assert [unit.source_entity_type for unit in episode_only.units] == ["podcast_episode"]
    assert episode_only.edges == []
    assert [unit.source_entity_type for unit in play_only.units] == ["play"]
    assert play_only.edges == []


def test_spotify_streaming_history_skip_reason_aggregates_music_and_podcasts(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T10:00:00Z",
                    "master_metadata_track_name": "Skipped Track",
                    "master_metadata_album_artist_name": "Artist A",
                    "spotify_track_uri": "spotify:track:skipped",
                    "ms_played": 100,
                    "reason_end": "fwdbtn",
                    "skipped": True,
                    "offline": False,
                },
                {
                    "ts": "2025-01-01T10:05:00Z",
                    "master_metadata_track_name": "Skipped Track",
                    "master_metadata_album_artist_name": "Artist A",
                    "spotify_track_uri": "spotify:track:skipped",
                    "ms_played": 200,
                    "reason_end": "fwdbtn",
                    "skipped": True,
                    "offline": False,
                },
                {
                    "ts": "2025-01-01T11:00:00Z",
                    "episode_name": "Offline Episode",
                    "episode_show_name": "Show A",
                    "spotify_episode_uri": "spotify:episode:offline",
                    "ms_played": 300,
                    "reason_end": "trackdone",
                    "skipped": False,
                    "offline": True,
                },
            ]
        ),
        encoding="utf-8",
    )

    result = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["skip_reason", "play"])

    skip_reasons = [unit for unit in result.units if unit.source_entity_type == "skip_reason"]
    plays = [unit for unit in result.units if unit.source_entity_type == "play"]
    assert all(unit.source_id.startswith("spotify_streaming_history:skip_reason:") for unit in skip_reasons)
    assert [unit.source_id for unit in skip_reasons] == [
        unit.source_id
        for unit in SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["skip_reason"]).units
    ]

    skipped = next(unit for unit in skip_reasons if unit.metadata["reason_end"] == "fwdbtn")
    assert skipped.metadata["skipped"] is True
    assert skipped.metadata["offline"] is False
    assert skipped.metadata["play_count"] == 2
    assert skipped.metadata["skipped_count"] == 2
    assert skipped.metadata["offline_count"] == 0
    assert skipped.metadata["total_ms_played"] == 300
    assert skipped.metadata["distinct_item_count"] == 1
    assert skipped.metadata["first_played_at"] == "2025-01-01T10:00:00+00:00"
    assert skipped.metadata["latest_played_at"] == "2025-01-01T10:05:00+00:00"
    assert skipped.metadata["play_source_ids"] == [plays[0].source_id, plays[1].source_id]

    podcast = next(unit for unit in skip_reasons if unit.metadata["reason_end"] == "trackdone")
    assert podcast.metadata["offline_count"] == 1
    assert podcast.metadata["distinct_item_count"] == 1
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "skip_reason_contains_play"]) == 3
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in result.edges)


def test_spotify_streaming_history_skip_reason_filtering(tmp_path):
    export = tmp_path / "endsong_0.json"
    export.write_text(
        json.dumps(
            [
                {
                    "ts": "2025-01-01T10:00:00Z",
                    "master_metadata_track_name": "Track",
                    "master_metadata_album_artist_name": "Artist",
                    "reason_end": "logout",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert "skip_reason" in SpotifyStreamingHistoryAdapter(path=str(export)).entity_types
    skip_reason_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["skip_reason"])
    play_only = SpotifyStreamingHistoryAdapter(path=str(export)).ingest(entity_types=["play"])

    assert [unit.source_entity_type for unit in skip_reason_only.units] == ["skip_reason"]
    assert skip_reason_only.edges == []
    assert [unit.source_entity_type for unit in play_only.units] == ["play"]
    assert play_only.edges == []


def test_spotify_streaming_history_adapter_is_registered():
    assert "spotify_streaming_history" in list_adapters()
    adapter = get_adapter("spotify_streaming_history", path="/tmp/spotify")
    assert adapter.name == "spotify_streaming_history"
