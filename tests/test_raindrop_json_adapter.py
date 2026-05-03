"""Tests for the Raindrop.io JSON bookmark export adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.raindrop_json import RaindropJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_get_adapter_returns_raindrop_json_instance(tmp_path):
    export = tmp_path / "raindrop.json"
    adapter = get_adapter("raindrop_json", path=str(export))

    assert isinstance(adapter, RaindropJsonAdapter)
    assert adapter.name == "raindrop_json"


def test_ingest_json_export_preserves_bookmark_metadata(tmp_path):
    export = tmp_path / "raindrop.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "_id": 123,
                        "title": "Graph Notes",
                        "link": "https://example.com/graph",
                        "excerpt": "A useful graph article.",
                        "note": "Remember to cite this.",
                        "tags": ["PKM", "graph", "pkm", "#Research"],
                        "collection": {"title": "Reading"},
                        "created": "2024-01-02T03:04:05.000Z",
                        "lastUpdate": "2024-01-03T04:05:06.000Z",
                        "type": "link",
                        "domain": "example.com",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = RaindropJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "raindrop_json"
    assert unit.source_id == "raindrop_json:123"
    assert unit.source_entity_type == "bookmark"
    assert unit.title == "Graph Notes"
    assert unit.content == (
        "Graph Notes\n"
        "URL: https://example.com/graph\n"
        "Description: A useful graph article.\n"
        "Notes: Remember to cite this.\n"
        "Collection: Reading\n"
        "Tags: graph, pkm, research"
    )
    assert unit.content_type == "artifact"
    assert unit.tags == ["graph", "pkm", "research"]
    assert unit.metadata == {
        "url": "https://example.com/graph",
        "description": "A useful graph article.",
        "excerpt": "A useful graph article.",
        "collection": "Reading",
        "notes": "Remember to cite this.",
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


def test_ingest_nested_exports_and_notes_arrays(tmp_path):
    export = tmp_path / "nested.json"
    export.write_text(
        json.dumps(
            {
                "collections": {
                    "archive": {
                        "raindrops": [
                            {
                                "id": "abc",
                                "name": "Nested Export",
                                "url": "https://example.com/nested",
                                "description": "Description text.",
                                "notes": [
                                    {"text": "First note"},
                                    {"body": "Second note"},
                                ],
                                "tag": "Beta; Alpha",
                                "folder": ["Archive", {"title": "To Read"}],
                                "created_at": "2024-02-01T00:00:00+09:00",
                            }
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = RaindropJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "raindrop_json:abc"
    assert unit.title == "Nested Export"
    assert unit.metadata["url"] == "https://example.com/nested"
    assert unit.metadata["description"] == "Description text."
    assert unit.metadata["collection"] == "Archive / To Read"
    assert unit.metadata["notes"] == "First note\nSecond note"
    assert unit.tags == ["alpha", "beta"]
    assert unit.created_at == datetime(2024, 1, 31, 15, 0, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 1, 31, 15, 0, 0, tzinfo=timezone.utc)


def test_invalid_records_are_skipped_predictably(tmp_path):
    export = tmp_path / "raindrop.json"
    export.write_text(
        json.dumps(
            [
                {"id": "missing-url", "title": "No URL"},
                {"id": "valid", "title": "Valid", "link": "https://example.com/valid"},
                "not a record",
            ]
        ),
        encoding="utf-8",
    )

    result = RaindropJsonAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == ["raindrop_json:valid"]


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

    first = RaindropJsonAdapter(path=str(export)).ingest().units[0]
    second = RaindropJsonAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("url:")
    assert first.tags == ["alpha", "zeta"]


def test_missing_malformed_and_filtered_ingests_return_empty_result(tmp_path):
    missing = tmp_path / "missing.json"
    malformed = tmp_path / "malformed.json"
    valid = tmp_path / "valid.json"
    malformed.write_text("{not json", encoding="utf-8")
    valid.write_text(
        json.dumps([{"id": "1", "title": "Filtered", "link": "https://example.com"}]),
        encoding="utf-8",
    )

    assert RaindropJsonAdapter(path=str(missing)).ingest().units == []
    assert RaindropJsonAdapter(path=str(malformed)).ingest().units == []
    assert RaindropJsonAdapter(path=str(valid)).ingest(entity_types=["article"]).units == []


def test_since_filter_uses_updated_or_created_datetime(tmp_path):
    export = tmp_path / "raindrop.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "old",
                    "title": "Old",
                    "link": "https://example.com/old",
                    "created": "2024-01-01T00:00:00Z",
                },
                {
                    "id": "new",
                    "title": "New",
                    "link": "https://example.com/new",
                    "created": "2024-01-03T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = RaindropJsonAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="raindrop_json",
            source_entity_type="bookmark",
            last_sync_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    )

    assert [unit.source_id for unit in result.units] == ["raindrop_json:new"]
