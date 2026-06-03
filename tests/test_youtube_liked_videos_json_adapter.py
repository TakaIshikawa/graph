from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.youtube_liked_videos_json import YouTubeLikedVideosJsonAdapter
from graph.types.models import SyncState


def test_youtube_liked_videos_json_ingests_flat_and_renderer_records_filters_and_registry(tmp_path):
    path = tmp_path / "likes.json"
    path.write_text(
        json.dumps(
            [
                {"videoId": "old111", "title": "Old", "channelTitle": "Old Channel", "likedAt": "2026-04-01T00:00:00Z"},
                {
                    "videoRenderer": {
                        "videoId": "new222",
                        "title": {"runs": [{"text": "New Renderer"}]},
                        "ownerText": {"runs": [{"text": "New Channel"}]},
                        "descriptionSnippet": {"runs": [{"text": "Snippet"}]},
                    },
                    "likedAt": "2026-05-03T10:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(source_project="youtube_liked_videos_json", source_entity_type="liked_video", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = YouTubeLikedVideosJsonAdapter(path=str(path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New Renderer"]
    unit = result.units[0]
    assert unit.source_id == "youtube_liked_videos_json:video:new222"
    assert unit.metadata["video_id"] == "new222"
    assert unit.metadata["url"] == "https://www.youtube.com/watch?v=new222"
    assert unit.metadata["channel"] == "New Channel"
    assert unit.metadata["description"] == "Snippet"
    assert YouTubeLikedVideosJsonAdapter(path=str(path)).ingest(entity_types=["playlist"]).units == []
    assert get_adapter("youtube_liked_videos_json", path=str(path)).name == "youtube_liked_videos_json"
