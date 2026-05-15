from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.rss_reader_starred_json import RssReaderStarredJsonAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_rss_reader_starred_json_ingests_item_metadata_content_and_registry(tmp_path):
    export = tmp_path / "starred.json"
    export.write_text(
        json.dumps(
            [
                {
                    "title": "Article",
                    "url": "https://example.com/article",
                    "feed": {"title": "Example Feed"},
                    "author": "Ada",
                    "summary": "Useful summary",
                    "tags": [{"name": "tech"}, "saved"],
                    "published_at": "2025-01-01T00:00:00Z",
                    "starred_at": "2025-01-02T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = RssReaderStarredJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.RSS_READER_STARRED_JSON
    assert unit.source_entity_type == "starred_feed_item"
    assert unit.metadata["url"] == "https://example.com/article"
    assert unit.metadata["feed_title"] == "Example Feed"
    assert unit.metadata["author"] == "Ada"
    assert unit.metadata["summary"] == "Useful summary"
    assert unit.metadata["tags"] == ["tech", "saved"]
    assert unit.metadata["published_at"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["starred_at"] == "2025-01-02T00:00:00+00:00"
    assert unit.metadata["source_file"] == "starred.json"
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert "Example Feed" in unit.content
    assert get_adapter("rss_reader_starred_json", path=str(export)).name == "rss_reader_starred_json"


def test_rss_reader_starred_json_wrappers_directory_since_and_filters(tmp_path):
    (tmp_path / "one.json").write_text(json.dumps({"entries": [{"title": "Old", "link": "https://example.com/old", "feed_title": "Feed", "saved_at": "2025-01-01T00:00:00Z"}]}), encoding="utf-8")
    (tmp_path / "two.json").write_text(json.dumps({"saved": [{"name": "New", "href": "https://example.com/new", "source": "Feed", "content_text": "Body", "labels": "a,b", "saved_at": "2025-01-03T00:00:00Z"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")

    adapter = RssReaderStarredJsonAdapter(path=str(tmp_path))
    sync = SyncState(source_project="rss_reader_starred_json", source_entity_type="starred_feed_item", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert first.units[0].metadata["tags"] == ["a", "b"]
    assert first.units[0].metadata["summary"] == "Body"
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["feed"]).units == []
