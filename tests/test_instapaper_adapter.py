from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.instapaper import InstapaperAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def unix_time(year: int, month: int, day: int) -> str:
    return str(int(datetime(year, month, day, tzinfo=timezone.utc).timestamp()))


def test_instapaper_csv_ingests_bookmarks(tmp_path):
    export = tmp_path / "instapaper.csv"
    export.write_text(
        "\n".join(
            [
                "url,title,description,folder,time,progress,starred,bookmark_id",
                f'https://example.com/article,Article Title,Great read,To Read,{unix_time(2025, 1, 5)},0.5,1,123',
            ]
        ),
        encoding="utf-8",
    )

    result = InstapaperAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.INSTAPAPER
    assert unit.source_id == "instapaper:123"
    assert unit.source_entity_type == "bookmark"
    assert unit.title == "Article Title"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Article Title" in unit.content
    assert "URL: https://example.com/article" in unit.content
    assert "Description: Great read" in unit.content
    assert "Folder: To Read" in unit.content
    assert unit.metadata == {
        "url": "https://example.com/article",
        "description": "Great read",
        "progress": "0.5",
        "starred": "1",
        "folder": "To Read",
        "hash": "123",
        "time": unix_time(2025, 1, 5),
    }
    assert unit.tags == []
    assert unit.created_at == datetime(2025, 1, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 5, tzinfo=timezone.utc)


def test_instapaper_json_ingests_bookmarks_from_list_or_dict_exports(tmp_path):
    export = tmp_path / "instapaper.json"
    export.write_text(
        json.dumps(
            {
                "bookmarks": [
                    {
                        "bookmark_id": "abc",
                        "url": "https://example.com/json",
                        "title": "JSON Bookmark",
                        "description": "JSON description",
                        "time": "2025-01-06T12:30:00Z",
                        "progress": "0.75",
                        "folder": "Archive",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = InstapaperAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "instapaper:abc"
    assert unit.title == "JSON Bookmark"
    assert unit.metadata["url"] == "https://example.com/json"
    assert unit.metadata["description"] == "JSON description"
    assert unit.metadata["progress"] == "0.75"
    assert unit.metadata["folder"] == "Archive"
    assert unit.metadata["time"] == "2025-01-06T12:30:00Z"
    assert unit.created_at == datetime(2025, 1, 6, 12, 30, tzinfo=timezone.utc)


def test_instapaper_json_uses_url_for_source_id_when_bookmark_id_is_missing(tmp_path):
    export = tmp_path / "instapaper.json"
    export.write_text(
        json.dumps(
            [
                {
                    "url": "https://example.com/url-only",
                    "title": "URL only",
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = InstapaperAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "url:https://example.com/url-only"


def test_instapaper_since_filters_bookmarks_not_newer_than_sync_state(tmp_path):
    export = tmp_path / "instapaper.csv"
    export.write_text(
        "\n".join(
            [
                "bookmark_id,url,title,time",
                f"old,https://example.com/old,Old,{unix_time(2025, 1, 1)}",
                f"equal,https://example.com/equal,Equal,{unix_time(2025, 1, 2)}",
                f"new,https://example.com/new,New,{unix_time(2025, 1, 3)}",
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="instapaper",
        source_entity_type="bookmark",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = InstapaperAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["instapaper:new"]


def test_instapaper_respects_entity_types(tmp_path):
    export = tmp_path / "instapaper.json"
    export.write_text(
        json.dumps([{"bookmark_id": "123", "url": "https://example.com"}]), encoding="utf-8"
    )

    result = InstapaperAdapter(path=str(export)).ingest(entity_types=["saved_item"])

    assert result.units == []
    assert result.edges == []


def test_instapaper_adapter_is_registered():
    assert "instapaper" in list_adapters()
    adapter = get_adapter("instapaper", path="/tmp/instapaper.json")
    assert adapter.name == "instapaper"
