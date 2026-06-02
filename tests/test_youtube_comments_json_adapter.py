import json

from graph.adapters.registry import get_adapter
from graph.adapters.youtube_comments_json import YoutubeCommentsJsonAdapter


def test_youtube_comments_json_ingests_wrapped_and_flat_records(tmp_path):
    path = tmp_path / "comments.json"
    path.write_text(json.dumps({"comments": [{"id": "c1", "text": "Great video", "video_id": "v1", "video_title": "Demo", "channel_name": "Graph", "url": "https://youtu.be/v1", "like_count": "3", "published_at": "2026-05-01T00:00:00Z"}]}), encoding="utf-8")

    unit = YoutubeCommentsJsonAdapter(str(path)).ingest().units[0]

    assert unit.source_id == "youtube_comments_json:c1"
    assert unit.metadata["video_id"] == "v1"
    assert unit.metadata["like_count"] == 3
    assert unit.tags == ["youtube", "comment"]


def test_youtube_comments_json_fallback_filter_and_registry(tmp_path):
    path = tmp_path / "comments.json"
    path.write_text(json.dumps([{"text": "No id", "video_id": "v2", "published_at": "2026-05-02T00:00:00Z"}, {}]), encoding="utf-8")

    result = YoutubeCommentsJsonAdapter(str(path)).ingest(entity_types=["comment"])

    assert len(result.units) == 1
    assert result.units[0].source_id.startswith("youtube_comments_json:")
    assert YoutubeCommentsJsonAdapter(str(path)).ingest(entity_types=["video"]).units == []
    assert get_adapter("youtube_comments_json", path=str(path)).name == "youtube_comments_json"
