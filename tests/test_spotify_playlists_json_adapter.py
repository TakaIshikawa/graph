from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.spotify_playlists_json import SpotifyPlaylistsJsonAdapter
from graph.types.models import SyncState


def test_spotify_playlists_json_ingests_tracks_filters_and_registry(tmp_path):
    path = tmp_path / "spotify.json"
    path.write_text(json.dumps({"playlists": [{"id": "pl1", "name": "Mix", "description": "Desc", "owner": {"display_name": "Me"}, "collaborative": False, "tracks": {"items": [{"added_at": "2026-04-01", "track": {"id": "old", "name": "Old", "artists": [{"name": "A"}]}}, {"added_at": "2026-05-03", "track": {"id": "new", "name": "New", "artists": [{"name": "B"}], "album": {"name": "Album"}, "uri": "spotify:track:new", "external_urls": {"spotify": "https://open.spotify.com/track/new"}}}]}}]}), encoding="utf-8")
    since = SyncState(source_project="spotify_playlists_json", source_entity_type="playlist_track", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = SpotifyPlaylistsJsonAdapter(path=str(path)).ingest(since=since)

    assert {unit.source_entity_type for unit in result.units} == {"playlist", "playlist_track"}
    track = [unit for unit in result.units if unit.source_entity_type == "playlist_track"][0]
    assert track.title == "New"
    assert track.metadata["artists"] == ["B"]
    assert track.metadata["album"] == "Album"
    assert len(result.edges) == 1
    assert SpotifyPlaylistsJsonAdapter(path=str(path)).ingest(entity_types=["playlist_track"]).edges == []
    assert get_adapter("spotify_playlists_json", path=str(path)).name == "spotify_playlists_json"
