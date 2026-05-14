from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.youtube_playlists_json import YouTubePlaylistsJsonAdapter
from graph.types.enums import EdgeRelation, SourceProject
from graph.types.models import SyncState


def test_youtube_playlists_json_ingests_playlists_videos_and_edges(tmp_path):
    export = tmp_path / "playlists.json"
    export.write_text(
        json.dumps(
            {
                "playlists": [
                    {
                        "playlistId": "PL1",
                        "title": "Watch Later",
                        "description": "Saved videos",
                        "privacyStatus": "private",
                        "videos": [
                            {
                                "videoId": "abc123",
                                "title": "Video One",
                                "channelTitle": "Channel A",
                                "url": "https://youtube.com/watch?v=abc123",
                                "position": 1,
                                "addedAt": "2026-05-01T10:00:00Z",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = YouTubePlaylistsJsonAdapter(path=str(export)).ingest()

    units = {unit.source_entity_type: unit for unit in result.units}
    assert set(units) == {"playlist", "playlist_video"}
    playlist = units["playlist"]
    video = units["playlist_video"]
    assert playlist.source_project == SourceProject.YOUTUBE_PLAYLISTS_JSON
    assert playlist.source_id == "youtube_playlists_json:playlist:PL1"
    assert playlist.metadata["privacy"] == "private"
    assert video.metadata["video_id"] == "abc123"
    assert video.metadata["channel"] == "Channel A"
    assert video.metadata["position"] == 1
    assert video.updated_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert len(result.edges) == 1
    assert result.edges[0].from_unit_id == playlist.source_id
    assert result.edges[0].to_unit_id == video.source_id
    assert result.edges[0].relation == EdgeRelation.CONTAINS


def test_youtube_playlists_json_filters_since_entities_missing_ids_and_malformed(tmp_path):
    (tmp_path / "good.json").write_text(
        json.dumps(
            [
                {
                    "name": "Playlist",
                    "items": [
                        {"title": "Old", "url": "https://youtu.be/old111", "added_at": "2026-04-30"},
                        {"title": "New", "url": "https://youtube.com/watch?v=new222", "added_at": "2026-05-03"},
                        {"title": "", "url": ""},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")
    since = SyncState(
        source_project="youtube_playlists_json",
        source_entity_type="playlist_video",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    videos_only = YouTubePlaylistsJsonAdapter(path=str(tmp_path)).ingest(since=since, entity_types=["playlist_video"])
    playlists_only = YouTubePlaylistsJsonAdapter(path=str(tmp_path)).ingest(entity_types=["playlist"])
    skipped = YouTubePlaylistsJsonAdapter(path=str(tmp_path)).ingest(entity_types=["watch"])

    assert [unit.title for unit in videos_only.units] == ["New"]
    assert videos_only.units[0].metadata["video_id"] == "new222"
    assert videos_only.edges == []
    assert [unit.source_entity_type for unit in playlists_only.units] == ["playlist"]
    assert playlists_only.edges == []
    assert skipped.units == []
    assert get_adapter("youtube_playlists_json", path=str(tmp_path)).name == "youtube_playlists_json"


def test_youtube_playlists_json_deduplicates_edges(tmp_path):
    export = tmp_path / "playlists.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "PL1",
                        "title": "Playlist",
                        "videos": [
                            {"id": "v1", "title": "Video", "position": 0, "addedAt": "2026-05-01"},
                            {"id": "v1", "title": "Video", "position": 0, "addedAt": "2026-05-01"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8-sig",
    )

    result = YouTubePlaylistsJsonAdapter(path=str(export)).ingest()

    assert len(result.edges) == 1
