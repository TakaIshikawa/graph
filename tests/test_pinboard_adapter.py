from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.pinboard import PinboardAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject


def test_pinboard_json_array_ingests_bookmarks(tmp_path):
    export = tmp_path / "pinboard.json"
    export.write_text(
        json.dumps(
            [
                {
                    "href": "https://example.com/agents",
                    "description": "Agent Notes",
                    "extended": "Useful notes",
                    "tags": "AI research #Saved",
                    "time": "2025-01-02T03:04:05Z",
                    "shared": "yes",
                    "toread": "no",
                    "hash": "abc123",
                }
            ]
        ),
        encoding="utf-8",
    )

    first = PinboardAdapter(path=str(export)).ingest()
    second = PinboardAdapter(path=str(export)).ingest()

    assert len(first.units) == 1
    unit = first.units[0]
    assert unit.source_project == SourceProject.PINBOARD
    assert unit.source_entity_type == "bookmark"
    assert unit.source_id == "pinboard:abc123"
    assert unit.source_id == second.units[0].source_id
    assert unit.title == "Agent Notes"
    assert unit.content_type == ContentType.ARTIFACT
    assert "URL: https://example.com/agents" in unit.content
    assert "Notes: Useful notes" in unit.content
    assert "Tags: ai, research, saved" in unit.content
    assert unit.metadata == {
        "url": "https://example.com/agents",
        "notes": "Useful notes",
        "hash": "abc123",
        "shared": "yes",
        "toread": "no",
        "time": "2025-01-02T03:04:05Z",
        "tags": ["ai", "research", "saved"],
    }
    assert unit.tags == ["ai", "research", "saved"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_pinboard_nested_export_objects_and_directory_are_ingested(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "nested" / "b.json"
    second.parent.mkdir()
    first.write_text(
        json.dumps(
            {
                "exported_at": "2025-01-01T00:00:00Z",
                "posts": [
                    {
                        "url": "https://example.com/url-field",
                        "title": "URL Field",
                        "notes": "Nested notes",
                        "tags": ["Knowledge", "knowledge", "Tools"],
                        "time": "2025-01-03T00:00:00+09:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "data": {
                    "bookmarks": [
                        {
                            "href": "https://example.com/second",
                            "description": "Second",
                            "time": "1735689600",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = PinboardAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "url:https://example.com/second",
        "url:https://example.com/url-field",
    ]
    url_field = result.units[1]
    assert url_field.title == "URL Field"
    assert url_field.metadata["notes"] == "Nested notes"
    assert url_field.tags == ["knowledge", "tools"]
    assert url_field.created_at == datetime(2025, 1, 2, 15, 0, tzinfo=timezone.utc)
    assert result.units[0].created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_pinboard_respects_entity_types(tmp_path):
    export = tmp_path / "pinboard.json"
    export.write_text(
        json.dumps([{"href": "https://example.com", "description": "Example"}]),
        encoding="utf-8",
    )

    result = PinboardAdapter(path=str(export)).ingest(entity_types=["saved_item"])

    assert result.units == []
    assert result.edges == []


def test_missing_pinboard_path_returns_empty_result(tmp_path):
    result = PinboardAdapter(path=str(tmp_path / "missing.json")).ingest()

    assert result.units == []
    assert result.edges == []


def test_pinboard_adapter_is_registered():
    assert "pinboard" in list_adapters()
    adapter = get_adapter("pinboard", path="/tmp/pinboard.json")
    assert adapter.name == "pinboard"
