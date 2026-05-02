"""Tests for the Raindrop.io bookmark export adapter."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from graph.adapters.raindrop import RaindropAdapter
from graph.adapters.registry import get_adapter


def test_get_adapter_returns_raindrop_instance(tmp_path):
    export = tmp_path / "raindrop.json"
    adapter = get_adapter("raindrop", path=str(export))

    assert isinstance(adapter, RaindropAdapter)
    assert adapter.name == "raindrop"


def test_ingest_json_list_export_preserves_bookmark_fields(tmp_path):
    export = tmp_path / "raindrop.json"
    export.write_text(
        json.dumps(
            [
                {
                    "_id": 123,
                    "title": "Graph Notes",
                    "link": "https://example.com/graph",
                    "excerpt": "A useful graph article.",
                    "tags": ["PKM", "graph", "pkm", "#Research"],
                    "collection": {"title": "Reading"},
                    "created": "2024-01-02T03:04:05.000Z",
                    "lastUpdate": "2024-01-03T04:05:06.000Z",
                    "type": "link",
                    "domain": "example.com",
                },
                {
                    "_id": 456,
                    "title": "No URL",
                    "excerpt": "Kept because it has a title.",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = RaindropAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == ["raindrop:123", "raindrop:456"]
    unit = result.units[0]
    assert unit.source_project == "raindrop"
    assert unit.source_entity_type == "bookmark"
    assert unit.title == "Graph Notes"
    assert unit.content == (
        "Graph Notes\n"
        "URL: https://example.com/graph\n"
        "Excerpt: A useful graph article.\n"
        "Collection: Reading\n"
        "Tags: graph, pkm, research"
    )
    assert unit.content_type == "artifact"
    assert unit.tags == ["graph", "pkm", "research"]
    assert unit.metadata == {
        "url": "https://example.com/graph",
        "excerpt": "A useful graph article.",
        "collection": "Reading",
        "created_at": "2024-01-02T03:04:05.000Z",
        "updated_at": "2024-01-03T04:05:06.000Z",
        "tags": ["graph", "pkm", "research"],
        "raindrop_id": "123",
        "type": "link",
        "domain": "example.com",
        "cover": "",
        "source_file": "raindrop.json",
    }
    assert unit.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 1, 3, 4, 5, 6, tzinfo=timezone.utc)


def test_ingest_object_export_with_items_array(tmp_path):
    export = tmp_path / "raindrop-items.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "abc",
                        "name": "Object Export",
                        "url": "https://example.com/object",
                        "description": "Description text.",
                        "tag": "Beta; Alpha",
                        "folder": "Archive",
                        "created_at": "2024-02-01T00:00:00+09:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = RaindropAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "raindrop:abc"
    assert unit.title == "Object Export"
    assert unit.metadata["url"] == "https://example.com/object"
    assert unit.metadata["excerpt"] == "Description text."
    assert unit.metadata["collection"] == "Archive"
    assert unit.tags == ["alpha", "beta"]
    assert unit.created_at == datetime(2024, 1, 31, 15, 0, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 1, 31, 15, 0, 0, tzinfo=timezone.utc)


def test_ingest_csv_export(tmp_path):
    export = tmp_path / "raindrop.csv"
    with export.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "title",
                "link",
                "excerpt",
                "tags",
                "collection",
                "created",
                "lastUpdate",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "csv-1",
                "title": "CSV Bookmark",
                "link": "https://example.com/csv",
                "excerpt": "CSV excerpt.",
                "tags": "Writing, Research | Writing",
                "collection": "Inbox",
                "created": "1710000000",
                "lastUpdate": "1710000100",
            }
        )

    result = RaindropAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "raindrop:csv-1"
    assert unit.metadata["url"] == "https://example.com/csv"
    assert unit.metadata["collection"] == "Inbox"
    assert unit.tags == ["research", "writing"]
    assert unit.created_at == datetime.fromtimestamp(1710000000, tz=timezone.utc)
    assert unit.updated_at == datetime.fromtimestamp(1710000100, tz=timezone.utc)


def test_missing_and_malformed_files_return_empty_result(tmp_path):
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not json", encoding="utf-8")

    assert RaindropAdapter(path=str(missing)).ingest().units == []
    assert RaindropAdapter(path=str(malformed)).ingest().units == []


def test_entity_types_filtering_returns_no_bookmarks_for_other_types(tmp_path):
    export = tmp_path / "raindrop.json"
    export.write_text(
        json.dumps([{"id": "1", "title": "Filtered", "link": "https://example.com"}]),
        encoding="utf-8",
    )

    result = RaindropAdapter(path=str(export)).ingest(entity_types=["article"])

    assert result.units == []
    assert result.edges == []


def test_url_source_id_fallback_is_stable(tmp_path):
    export = tmp_path / "raindrop.json"
    export.write_text(
        json.dumps(
            [
                {
                    "title": "No Export ID",
                    "link": "https://example.com/stable",
                    "tags": [{"name": "Zeta"}, {"tag": "alpha"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = RaindropAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == "url:https://example.com/stable"
    assert result.units[0].tags == ["alpha", "zeta"]
