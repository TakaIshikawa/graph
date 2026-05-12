from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.youtube_watch_history_json import YouTubeWatchHistoryJsonAdapter
from graph.types.models import SyncState


def test_youtube_watch_history_json_ingests_list_payload(tmp_path):
    export = tmp_path / "watch-history.json"
    export.write_text(
        """[
          {
            "title": "Watched A Useful Video",
            "titleUrl": "https://www.youtube.com/watch?v=abc",
            "time": "2026-05-01T12:30:00Z",
            "products": ["YouTube"],
            "subtitles": [{"name": "Useful Channel", "url": "https://www.youtube.com/channel/1"}]
          }
        ]""",
        encoding="utf-8",
    )

    result = YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_entity_type == "watch"
    assert unit.source_id == YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest().units[0].source_id
    assert unit.metadata["title_url"] == "https://www.youtube.com/watch?v=abc"
    assert unit.metadata["channel_name"] == "Useful Channel"
    assert unit.metadata["channel_url"] == "https://www.youtube.com/channel/1"
    assert unit.metadata["products"] == ["YouTube"]
    assert "Channel: Useful Channel" in unit.content
    assert unit.tags == ["youtube", "watch"]
    assert unit.updated_at == datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def test_youtube_watch_history_json_ingests_directory_and_skips_bad_files(tmp_path):
    good = tmp_path / "one.json"
    bad = tmp_path / "bad.json"
    good.write_text(
        """{"items": [{"title": "Watched One", "titleUrl": "https://youtu.be/one", "time": "2026-05-02T00:00:00Z"}]}""",
        encoding="utf-8",
    )
    bad.write_text("{not json", encoding="utf-8")

    result = YouTubeWatchHistoryJsonAdapter(path=str(tmp_path)).ingest()

    assert [unit.title for unit in result.units] == ["Watched One"]
    assert result.units[0].metadata["source_file"] == "one.json"


def test_youtube_watch_history_json_uses_title_time_fallback_id(tmp_path):
    export = tmp_path / "watch-history.json"
    export.write_text(
        """[
          {"title": "Watched No URL", "time": "2026-05-03T10:00:00Z"},
          {"title": "Watched No URL", "time": "2026-05-03T10:00:00Z"}
        ]""",
        encoding="utf-8",
    )

    result = YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    assert result.units[0].source_id == result.units[1].source_id
    assert result.units[0].source_id.startswith("youtube_watch_history_json:")


def test_youtube_watch_history_json_filters_since_and_entity_types(tmp_path):
    export = tmp_path / "watch-history.json"
    export.write_text(
        """[
          {"title": "Old", "time": "2026-05-01T00:00:00Z"},
          {"title": "New", "time": "2026-05-03T00:00:00Z"}
        ]""",
        encoding="utf-8",
    )
    since = SyncState(source_project="youtube_watch_history_json", source_entity_type="watch", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest(since=since)
    skipped = YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest(entity_types=["video"])

    assert [unit.title for unit in result.units] == ["New"]
    assert skipped.units == []


def test_youtube_watch_history_json_channel_aggregates_and_edges(tmp_path):
    export = tmp_path / "watch-history.json"
    export.write_text(
        """[
          {"title": "Watched One", "titleUrl": "https://youtu.be/1", "time": "2026-05-01T00:00:00Z", "subtitles": [{"name": "Useful Channel", "url": "https://www.youtube.com/channel/1"}]},
          {"title": "Watched Two", "titleUrl": "https://youtu.be/2", "time": "2026-05-02T00:00:00Z", "subtitles": [{"name": "Useful Channel", "url": "https://www.youtube.com/channel/1"}]}
        ]""",
        encoding="utf-8",
    )

    result = YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest(entity_types=["channel", "watch"])
    channel = next(unit for unit in result.units if unit.source_entity_type == "channel")
    watches = [unit for unit in result.units if unit.source_entity_type == "watch"]

    assert YouTubeWatchHistoryJsonAdapter().entity_types == ["channel", "watch"]
    assert channel.metadata["channel"] == "Useful Channel"
    assert channel.metadata["channel_url"] == "https://www.youtube.com/channel/1"
    assert channel.metadata["watch_count"] == 2
    assert channel.metadata["video_source_ids"] == [watch.source_id for watch in watches]
    assert channel.metadata["watched_video_source_ids"] == [watch.source_id for watch in watches]
    assert channel.metadata["first_watched_at"] == "2026-05-01T00:00:00+00:00"
    assert channel.metadata["latest_watched_at"] == "2026-05-02T00:00:00+00:00"
    assert len(result.edges) == 2
    assert all(edge.from_unit_id == channel.source_id for edge in result.edges)


def test_youtube_watch_history_json_channel_dedupes_by_url_then_name_and_filters_edges(tmp_path):
    export = tmp_path / "watch-history.json"
    export.write_text(
        """[
          {"title": "Watched One", "titleUrl": "https://youtu.be/1", "time": "2026-05-01T00:00:00Z", "subtitles": [{"name": "First Name", "url": "https://www.youtube.com/channel/1"}]},
          {"title": "Watched Two", "titleUrl": "https://youtu.be/2", "time": "2026-05-02T00:00:00Z", "subtitles": [{"name": "Second Name", "url": "https://www.youtube.com/channel/1"}]},
          {"title": "Watched Three", "titleUrl": "https://youtu.be/3", "time": "2026-05-03T00:00:00Z", "subtitles": [{"name": "Name Only"}]},
          {"title": "Watched Four", "titleUrl": "https://youtu.be/4", "time": "2026-05-04T00:00:00Z", "subtitles": [{"name": " name only "}]}
        ]""",
        encoding="utf-8",
    )

    result = YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest(entity_types=["channel", "watch"])
    channel_only = YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest(entity_types=["channel"])
    watch_only = YouTubeWatchHistoryJsonAdapter(path=str(export)).ingest(entity_types=["watch"])

    channels = [unit for unit in result.units if unit.source_entity_type == "channel"]
    assert len(channels) == 2
    assert sorted(channel.metadata["watch_count"] for channel in channels) == [2, 2]
    assert all(channel.source_id.startswith("youtube_watch_history_json:channel:") for channel in channels)
    assert len(result.edges) == 4
    assert channel_only.edges == []
    assert watch_only.edges == []
