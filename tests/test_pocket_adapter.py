from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.pocket import PocketAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def unix_time(year: int, month: int, day: int) -> str:
    return str(int(datetime(year, month, day, tzinfo=timezone.utc).timestamp()))


def test_pocket_csv_ingests_saved_items(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "\n".join(
            [
                "item_id,given_url,resolved_url,given_title,resolved_title,excerpt,status,time_added,time_updated,tags",
                f'123,https://example.com/original,https://example.com/final,Original,Resolved Title,Short excerpt,0,{unix_time(2025, 1, 1)},{unix_time(2025, 1, 3)},"AI, Reading List"',
            ]
        ),
        encoding="utf-8",
    )

    result = PocketAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.POCKET
    assert unit.source_id == "pocket:123"
    assert unit.source_entity_type == "saved_item"
    assert unit.title == "Resolved Title"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Resolved Title" in unit.content
    assert "URL: https://example.com/final" in unit.content
    assert "Excerpt: Short excerpt" in unit.content
    assert "Tags: ai, reading list" in unit.content
    assert unit.metadata == {
        "url": "https://example.com/final",
        "excerpt": "Short excerpt",
        "status": "0",
        "time_added": unix_time(2025, 1, 1),
        "time_updated": unix_time(2025, 1, 3),
        "tags": ["ai", "reading list"],
    }
    assert unit.tags == ["ai", "reading list"]
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 3, tzinfo=timezone.utc)


def test_pocket_json_ingests_saved_items_from_list_or_dict_exports(tmp_path):
    export = tmp_path / "pocket.json"
    export.write_text(
        json.dumps(
            {
                "list": {
                    "abc": {
                        "item_id": "abc",
                        "given_url": "https://example.com/json",
                        "given_title": "JSON Item",
                        "excerpt": "JSON excerpt",
                        "status": "1",
                        "time_added": "2025-01-04T00:00:00Z",
                        "time_updated": "2025-01-05T00:00:00Z",
                        "tags": {
                            "Knowledge": {"tag": "Knowledge"},
                            "Read Later": {"tag": "Read Later"},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    unit = PocketAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "pocket:abc"
    assert unit.title == "JSON Item"
    assert unit.metadata["url"] == "https://example.com/json"
    assert unit.metadata["excerpt"] == "JSON excerpt"
    assert unit.metadata["status"] == "1"
    assert unit.metadata["time_added"] == "2025-01-04T00:00:00Z"
    assert unit.metadata["time_updated"] == "2025-01-05T00:00:00Z"
    assert unit.tags == ["knowledge", "read later"]


def test_pocket_json_uses_url_for_source_id_when_item_id_is_missing(tmp_path):
    export = tmp_path / "pocket.json"
    export.write_text(
        json.dumps(
            [
                {
                    "url": "https://example.com/url-only",
                    "title": "URL only",
                    "tags": ["#Saved", "saved"],
                }
            ]
        ),
        encoding="utf-8",
    )

    unit = PocketAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "url:https://example.com/url-only"
    assert unit.tags == ["saved"]


def test_pocket_since_filters_items_not_newer_than_sync_state(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "\n".join(
            [
                "item_id,given_url,given_title,time_added,time_updated,tags",
                f"old,https://example.com/old,Old,{unix_time(2025, 1, 1)},,old",
                f"equal,https://example.com/equal,Equal,{unix_time(2025, 1, 1)},{unix_time(2025, 1, 2)},equal",
                f"new,https://example.com/new,New,{unix_time(2025, 1, 1)},{unix_time(2025, 1, 3)},new",
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="pocket",
        source_entity_type="saved_item",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = PocketAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["pocket:new"]


def test_pocket_respects_entity_types(tmp_path):
    export = tmp_path / "pocket.json"
    export.write_text(json.dumps([{"item_id": "123", "given_url": "https://example.com"}]), encoding="utf-8")

    result = PocketAdapter(path=str(export)).ingest(entity_types=["bookmark"])

    assert result.units == []
    assert result.edges == []


def test_pocket_adapter_is_registered():
    assert "pocket" in list_adapters()
    adapter = get_adapter("pocket", path="/tmp/pocket.json")
    assert adapter.name == "pocket"
