from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.spotify_saved_tracks_json import SpotifySavedTracksJsonAdapter
from graph.types.models import SyncState


def test_spotify_saved_tracks_json_ingests_api_items_and_registry(tmp_path):
    export = tmp_path / "saved.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "added_at": "2026-05-01T10:00:00Z",
                        "track": {
                            "id": "track-1",
                            "name": "Song A",
                            "artists": [{"id": "artist-1", "name": "Ada"}, {"name": "Grace"}],
                            "album": {"id": "album-1", "name": "Album A", "release_date": "2025-01-02"},
                            "duration_ms": 123456,
                            "popularity": 87,
                            "explicit": True,
                            "preview_url": "https://spotify.test/preview.mp3",
                            "external_urls": {"spotify": "https://open.spotify.com/track/track-1"},
                            "external_ids": {"isrc": "USABC1234567"},
                            "uri": "spotify:track:track-1",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = SpotifySavedTracksJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "spotify_saved_tracks_json"
    assert unit.source_id == "spotify_saved_tracks_json:track-1"
    assert unit.source_entity_type == "saved_track"
    assert unit.title == "Song A"
    assert unit.metadata["artists"] == ["Ada", "Grace"]
    assert unit.metadata["album"] == "Album A"
    assert unit.metadata["album_release_date"] == "2025-01-02"
    assert unit.metadata["duration_ms"] == 123456
    assert unit.metadata["popularity"] == 87
    assert unit.metadata["explicit"] is True
    assert unit.metadata["preview_url"] == "https://spotify.test/preview.mp3"
    assert unit.metadata["external_url"] == "https://open.spotify.com/track/track-1"
    assert unit.metadata["isrc"] == "USABC1234567"
    assert unit.metadata["source_file"] == "saved.json"
    assert unit.metadata["source_row"] == 1
    assert unit.created_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == unit.created_at
    assert isinstance(get_adapter("spotify_saved_tracks_json"), SpotifySavedTracksJsonAdapter)


def test_spotify_saved_tracks_json_accepts_saved_tracks_wrapper_and_flat_records(tmp_path):
    export = tmp_path / "flat.json"
    export.write_text(
        json.dumps(
            {
                "saved_tracks": [
                    {
                        "track_id": "track-2",
                        "track_name": "Song B",
                        "artist": "Ada, Grace",
                        "album_name": "Album B",
                        "release_date": "2024-03-04",
                        "added_at": "2026-05-02",
                        "updated_at": "2026-05-03T01:02:03Z",
                        "duration_ms": "321000",
                        "popularity": "55",
                        "explicit": "false",
                        "url": "https://open.spotify.com/track/track-2",
                        "isrc": "USABC7654321",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = SpotifySavedTracksJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "spotify_saved_tracks_json:track-2"
    assert unit.metadata["artists"] == ["Ada", "Grace"]
    assert unit.metadata["album"] == "Album B"
    assert unit.metadata["album_release_date"] == "2024-03-04"
    assert unit.metadata["duration_ms"] == 321000
    assert unit.metadata["popularity"] == 55
    assert unit.metadata["explicit"] is False
    assert unit.updated_at == datetime(2026, 5, 3, 1, 2, 3, tzinfo=timezone.utc)


def test_spotify_saved_tracks_json_filters_since_entity_types_and_uses_digest_fallback(tmp_path):
    export = tmp_path / "saved.json"
    export.write_text(
        json.dumps(
            [
                {"added_at": "2026-05-01T00:00:00Z", "track": {"name": "Old", "artists": [{"name": "Ada"}], "external_ids": {"isrc": "OLD"}}},
                {"updated_at": "2026-05-03T00:00:00Z", "track": {"name": "New", "artists": [{"name": "Ada"}], "external_ids": {"isrc": "NEW"}}},
            ]
        ),
        encoding="utf-8",
    )
    adapter = SpotifySavedTracksJsonAdapter(path=str(export))
    sync = SyncState(source_project="spotify_saved_tracks_json", source_entity_type="saved_track", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert first.units[0].source_id.startswith("spotify_saved_tracks_json:")
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["track"]).units == []
