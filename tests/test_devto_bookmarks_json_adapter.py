from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.devto_bookmarks_json import DevtoBookmarksJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_devto_bookmarks_json_ingests_article_array_and_prefers_canonical_url(tmp_path):
    export = tmp_path / "bookmarks.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": 101,
                    "title": "Build search with Python",
                    "url": "https://dev.to/example/build-search-with-python-temp",
                    "canonical_url": "https://example.com/build-search-with-python",
                    "user": {"username": "example"},
                    "tags": ["python", "search"],
                    "published_at": "2025-01-01T00:00:00Z",
                    "saved_at": "2025-01-02T03:04:05Z",
                    "positive_reactions_count": 42,
                    "reading_time_minutes": 7,
                }
            ]
        ),
        encoding="utf-8",
    )

    result = DevtoBookmarksJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.DEVTO_BOOKMARKS_JSON
    assert unit.source_entity_type == "article_bookmark"
    assert unit.title == "Build search with Python"
    assert unit.metadata["url"] == "https://example.com/build-search-with-python"
    assert unit.metadata["author_username"] == "example"
    assert unit.metadata["tags"] == ["python", "search"]
    assert unit.metadata["positive_reactions_count"] == 42
    assert unit.metadata["reading_time_minutes"] == 7
    assert unit.metadata["published_at"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["saved_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert "URL: https://example.com/build-search-with-python" in unit.content


def test_devto_bookmarks_json_supports_wrapped_payloads_tag_strings_and_missing_counters(tmp_path):
    export = tmp_path / "saved.json"
    export.write_text(
        json.dumps(
            {
                "articles": [
                    {"id": "1", "title": "Old article", "path": "/old/article", "tag_list": "python, old", "savedAt": "2025-01-01"},
                    {"id": "2", "title": "New article", "path": "/new/article", "tag_list": "python, json", "author": {"username": "writer"}, "savedAt": "2025-01-03"},
                ]
            }
        ),
        encoding="utf-8-sig",
    )
    since = SyncState(source_project="devto_bookmarks_json", source_entity_type="article_bookmark", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))

    result = DevtoBookmarksJsonAdapter(path=str(export)).ingest(since=since)
    skipped = DevtoBookmarksJsonAdapter(path=str(export)).ingest(entity_types=["comment"])

    assert [unit.title for unit in result.units] == ["New article"]
    assert result.units[0].metadata["url"] == "https://dev.to/new/article"
    assert result.units[0].metadata["tags"] == ["python", "json"]
    assert result.units[0].metadata["author_username"] == "writer"
    assert "positive_reactions_count" not in result.units[0].metadata
    assert "reading_time_minutes" not in result.units[0].metadata
    assert skipped.units == []


def test_devto_bookmarks_json_supports_items_key_and_registry(tmp_path):
    export = tmp_path / "items.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "title": "Items wrapper",
                        "url": "https://dev.to/example/items-wrapper",
                        "tags": "devto|bookmarks",
                        "bookmarked_at": "2025-01-01T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    first_result = DevtoBookmarksJsonAdapter(path=str(export)).ingest()
    second_result = DevtoBookmarksJsonAdapter(path=str(export)).ingest()

    assert first_result.units[0].metadata["tags"] == ["devto", "bookmarks"]
    assert first_result.units[0].source_id == second_result.units[0].source_id
    assert get_adapter("devto_bookmarks_json", path=str(export)).name == "devto_bookmarks_json"
