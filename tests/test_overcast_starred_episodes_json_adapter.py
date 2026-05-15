from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.overcast_starred_episodes_json import OvercastStarredEpisodesJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_overcast_starred_episodes_json_ingests_nested_metadata_and_registry(tmp_path):
    export = tmp_path / "overcast.json"
    export.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "podcast_title": "Tech Show",
                        "episode_title": "Episode One",
                        "episode_url": "https://overcast.fm/+abc",
                        "audio_url": "https://cdn.example.com/abc.mp3",
                        "description": "Useful episode",
                        "starred_at": "2025-01-02T03:04:05Z",
                        "published_at": "2025-01-01T00:00:00Z",
                        "duration": 3600,
                        "progress": "120",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = OvercastStarredEpisodesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.OVERCAST_STARRED_EPISODES_JSON
    assert unit.source_entity_type == "starred_episode"
    assert unit.metadata["podcast"] == "Tech Show"
    assert unit.metadata["episode_url"] == "https://overcast.fm/+abc"
    assert unit.metadata["audio_url"] == "https://cdn.example.com/abc.mp3"
    assert unit.metadata["duration"] == 3600
    assert unit.metadata["progress"] == 120
    assert unit.metadata["source_file"] == "overcast.json"
    assert unit.metadata["record"]["episode_title"] == "Episode One"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert get_adapter("overcast_starred_episodes_json", path=str(export)).name == "overcast_starred_episodes_json"


def test_overcast_starred_episodes_json_arrays_bad_files_since_and_filters(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps([{"title": "Old", "url": "https://example.com/old", "starred_at": "2025-01-01T00:00:00Z"}]), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps({"starred": [{"title": "New", "url": "https://example.com/new", "starred_at": "2025-01-03T00:00:00Z"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")

    adapter = OvercastStarredEpisodesJsonAdapter(path=str(tmp_path))
    sync = SyncState(source_project="overcast_starred_episodes_json", source_entity_type="starred_episode", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["episode"]).units == []
