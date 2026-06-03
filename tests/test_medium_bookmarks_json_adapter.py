from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.medium_bookmarks_json import MediumBookmarksJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_medium_bookmarks_json_ingests_nested_bom_metadata_and_registry(tmp_path):
    export = tmp_path / "medium.json"
    export.write_text(
        "\ufeff"
        + json.dumps(
            {
                "bookmarks": [
                    {
                        "title": "Useful Article",
                        "subtitle": "A short guide",
                        "author": {"name": "Ada"},
                        "publication": "Engineering",
                        "url": "https://medium.com/p/useful",
                        "tags": [{"name": "python"}, "data"],
                        "claps": "12",
                        "responses": 3,
                        "saved_at": "2025-01-02T03:04:05Z",
                        "published_at": "2025-01-01T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MediumBookmarksJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MEDIUM_BOOKMARKS_JSON
    assert unit.source_entity_type == "article_bookmark"
    assert unit.metadata["title"] == "Useful Article"
    assert unit.metadata["author"] == "Ada"
    assert unit.metadata["publication"] == "Engineering"
    assert unit.metadata["url"] == "https://medium.com/p/useful"
    assert unit.metadata["tags"] == ["python", "data"]
    assert unit.metadata["claps"] == 12
    assert unit.metadata["responses"] == 3
    assert unit.metadata["source_file"] == "medium.json"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert get_adapter("medium_bookmarks_json", path=str(export)).name == "medium_bookmarks_json"


def test_medium_bookmarks_json_skips_missing_identity_bad_files_since_and_filters(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps([{"title": "Old", "url": "https://example.com/old", "updated_at": "2025-01-01T00:00:00Z"}]), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps({"items": [{"title": "New", "url": "https://example.com/new", "updated_at": "2025-01-03T00:00:00Z"}, {"subtitle": "No identity"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")

    adapter = MediumBookmarksJsonAdapter(path=str(tmp_path))
    sync = SyncState(source_project="medium_bookmarks_json", source_entity_type="article_bookmark", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["article"]).units == []


def test_medium_bookmarks_json_accepts_nested_missing_people_and_deduplicates_urls(tmp_path):
    export = tmp_path / "nested.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "story": {
                            "title": "Nested Story",
                            "preview": "Nested preview",
                            "url": "https://medium.com/p/nested",
                            "topics": ["engineering"],
                            "readingTime": "5",
                        },
                        "saved_at": "2025-01-04T00:00:00Z",
                    },
                    {
                        "title": "Duplicate Story",
                        "url": "https://medium.com/p/nested",
                        "saved_at": "2025-01-05T00:00:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MediumBookmarksJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["title"] == "Duplicate Story"
    assert unit.metadata["url"] == "https://medium.com/p/nested"
    assert "author" not in unit.metadata
    assert "publication" not in unit.metadata
    assert unit.source_id == MediumBookmarksJsonAdapter(path=str(export)).ingest().units[0].source_id
