from __future__ import annotations

from graph.adapters.registry import get_adapter
from graph.adapters.youtube_playlists_csv import YoutubePlaylistsCsvAdapter


def test_youtube_playlists_csv_ingests_playlist_items(tmp_path):
    export = tmp_path / "youtube.csv"
    export.write_text("Playlist ID,Playlist Name,Video ID,Video Title,Channel,Video URL,Added At,Position,Description,Privacy\npl1,Research,v1,Talk,Ada,https://youtu.be/v1,2026-05-01T00:00:00Z,2,Deep dive,public\n", encoding="utf-8")

    unit = YoutubePlaylistsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "playlist_item"
    assert unit.metadata["playlist_id"] == "pl1"
    assert unit.metadata["video_id"] == "v1"
    assert unit.metadata["position"] == 2
    assert "Playlist: Research" in unit.content


def test_youtube_playlists_csv_falls_back_stably(tmp_path):
    export = tmp_path / "youtube.csv"
    export.write_text("Playlist Name,Video Title\nResearch,Talk\n", encoding="utf-8")

    first = YoutubePlaylistsCsvAdapter(path=str(export)).ingest().units[0]
    second = YoutubePlaylistsCsvAdapter(path=str(export)).ingest().units[0]
    assert first.source_id == second.source_id


def test_youtube_playlists_csv_is_registered():
    assert isinstance(get_adapter("youtube-playlists-csv"), YoutubePlaylistsCsvAdapter)
