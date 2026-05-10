from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.wallabag import WallabagAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def unix_time(year: int, month: int, day: int) -> str:
    return str(int(datetime(year, month, day, tzinfo=timezone.utc).timestamp()))


def test_wallabag_json_ingests_articles(tmp_path):
    export = tmp_path / "wallabag.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "123",
                    "title": "Article Title",
                    "url": "https://example.com/article",
                    "content": "<p>Article content here</p>",
                    "tags": [{"label": "research"}, {"label": "important"}],
                    "reading_time": "5",
                    "is_archived": "1",
                    "is_starred": "0",
                    "created_at": "2025-01-10T12:00:00Z",
                    "updated_at": "2025-01-11T14:30:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = WallabagAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.WALLABAG
    assert unit.source_id == "wallabag:123"
    assert unit.source_entity_type == "article"
    assert unit.title == "Article Title"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Article Title" in unit.content
    assert "URL: https://example.com/article" in unit.content
    assert "Content: <p>Article content here</p>" in unit.content
    assert unit.metadata == {
        "url": "https://example.com/article",
        "content": "<p>Article content here</p>",
        "reading_time": "5",
        "archived": "1",
        "starred": "0",
        "tags": ["research", "important"],
        "created_at": "2025-01-10T12:00:00Z",
        "updated_at": "2025-01-11T14:30:00Z",
        "wallabag_id": "123",
    }
    assert unit.tags == ["research", "important"]
    assert unit.created_at == datetime(2025, 1, 10, 12, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 11, 14, 30, tzinfo=timezone.utc)


def test_wallabag_json_handles_nested_entries(tmp_path):
    export = tmp_path / "wallabag.json"
    export.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "id": "456",
                        "title": "Nested Article",
                        "url": "https://example.com/nested",
                        "created_at": "2025-01-12T10:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = WallabagAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == "wallabag:456"
    assert result.units[0].title == "Nested Article"


def test_wallabag_json_handles_missing_fields(tmp_path):
    export = tmp_path / "wallabag.json"
    export.write_text(
        json.dumps(
            [
                {
                    "url": "https://example.com/minimal",
                    "title": "Minimal Article",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = WallabagAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "url:https://example.com/minimal"
    assert unit.title == "Minimal Article"
    assert unit.metadata["reading_time"] == ""
    assert unit.metadata["archived"] == ""
    assert unit.metadata["starred"] == ""
    assert unit.tags == []


def test_wallabag_filters_by_sync_state(tmp_path):
    export = tmp_path / "wallabag.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "old",
                    "url": "https://example.com/old",
                    "title": "Old",
                    "updated_at": "2025-01-01T00:00:00Z",
                },
                {
                    "id": "equal",
                    "url": "https://example.com/equal",
                    "title": "Equal",
                    "updated_at": "2025-01-02T00:00:00Z",
                },
                {
                    "id": "new",
                    "url": "https://example.com/new",
                    "title": "New",
                    "updated_at": "2025-01-03T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="wallabag",
        source_entity_type="article",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = WallabagAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["wallabag:new"]


def test_wallabag_respects_entity_types(tmp_path):
    export = tmp_path / "wallabag.json"
    export.write_text(
        json.dumps([{"url": "https://example.com", "title": "Test"}]), encoding="utf-8"
    )

    result = WallabagAdapter(path=str(export)).ingest(entity_types=["bookmark"])

    assert result.units == []


def test_wallabag_handles_malformed_json(tmp_path):
    export = tmp_path / "wallabag.json"
    export.write_text("not valid json", encoding="utf-8")

    result = WallabagAdapter(path=str(export)).ingest()

    assert result.units == []
    assert result.edges == []


def test_wallabag_handles_missing_file():
    result = WallabagAdapter(path="/nonexistent/file.json").ingest()
    assert result.units == []
    assert result.edges == []


def test_wallabag_adapter_is_registered():
    assert "wallabag" in list_adapters()
    adapter = get_adapter("wallabag", path="/tmp/wallabag.json")
    assert adapter.name == "wallabag"
