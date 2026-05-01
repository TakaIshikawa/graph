from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.readwise import ReadwiseAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_readwise_ingests_top_level_highlight_list(tmp_path):
    export = tmp_path / "readwise.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": 2,
                    "text": "Second highlight",
                    "note": "Second note",
                    "title": "Second Book",
                    "author": "Second Author",
                    "book_id": 202,
                    "source_url": "https://example.com/second",
                    "location": 12,
                    "location_type": "page",
                    "tags": [{"name": "Research"}, {"name": "PKM"}],
                    "created_at": "2025-01-02T10:00:00Z",
                    "updated_at": "2025-01-03T10:00:00Z",
                    "highlighted_at": "2025-01-02T11:00:00Z",
                },
                {
                    "id": 1,
                    "highlighted_text": "First highlight",
                    "book": {
                        "id": 101,
                        "title": "First Book",
                        "author": "First Author",
                        "source_url": "https://example.com/first",
                    },
                    "location": "33",
                    "tags": "ideas, #reading",
                    "created_at": "2025-01-01T10:00:00Z",
                    "highlighted_at": "2025-01-01T11:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = ReadwiseAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in result.units] == ["readwise:1", "readwise:2"]
    unit = result.units[0]
    assert unit.source_project == SourceProject.READWISE
    assert unit.source_entity_type == "highlight"
    assert unit.title == "First Book"
    assert unit.content_type == ContentType.INSIGHT
    assert "First highlight" in unit.content
    assert "Author: First Author" in unit.content
    assert "URL: https://example.com/first" in unit.content
    assert "Location: 33" in unit.content
    assert unit.metadata["book_id"] == "101"
    assert unit.metadata["source_url"] == "https://example.com/first"
    assert unit.metadata["text"] == "First highlight"
    assert unit.metadata["tags"] == ["ideas", "reading"]
    assert unit.tags == ["ideas", "reading"]
    assert unit.created_at == datetime(2025, 1, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 1, 11, tzinfo=timezone.utc)


def test_readwise_ingests_results_and_highlights_wrappers(tmp_path):
    export = tmp_path / "readwise.json"
    export.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "title": "Wrapped Book",
                        "author": "Wrapped Author",
                        "book_id": "book-1",
                        "highlights": [
                            {
                                "id": "wrapped-1",
                                "text": "Nested highlight",
                                "note": "Nested note",
                                "location": "chapter 2",
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ReadwiseAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "readwise:wrapped-1"
    assert unit.title == "Wrapped Book"
    assert unit.metadata["author"] == "Wrapped Author"
    assert unit.metadata["book_id"] == "book-1"
    assert unit.metadata["note"] == "Nested note"


def test_readwise_directory_discovers_json_and_skips_bad_records(tmp_path):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    (tmp_path / "notes.txt").write_text(
        json.dumps([{"id": "ignored", "text": "Not JSON suffix"}]), encoding="utf-8"
    )
    (tmp_path / "non_highlight.json").write_text(
        json.dumps({"results": [{"id": "book-only", "title": "Book only"}]}),
        encoding="utf-8",
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "highlights.json").write_text(
        json.dumps(
            {
                "highlights": [
                    {"id": "a", "text": "Alpha"},
                    "not a dict",
                    {"id": "b", "note": "Note-only highlight"},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ReadwiseAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == ["readwise:a", "readwise:b"]


def test_readwise_uses_stable_hash_source_id_when_id_is_missing(tmp_path):
    export = tmp_path / "readwise.json"
    export.write_text(
        json.dumps(
            [
                {
                    "text": "Stable highlight",
                    "title": "Stable Book",
                    "book_id": "book-1",
                    "location": "42",
                    "highlighted_at": "2025-01-01T00:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    first = ReadwiseAdapter(path=str(export)).ingest().units[0]
    second = ReadwiseAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("readwise:")


def test_readwise_since_filters_using_event_timestamps(tmp_path):
    export = tmp_path / "readwise.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "old-created",
                    "text": "Old created",
                    "created_at": "2025-01-01T00:00:00Z",
                },
                {
                    "id": "new-highlighted",
                    "text": "New highlighted",
                    "highlighted_at": "2025-01-03T00:00:00Z",
                },
                {
                    "id": "new-updated",
                    "text": "New updated",
                    "updated_at": "2025-01-04T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="readwise",
        source_entity_type="highlight",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = ReadwiseAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == [
        "readwise:new-highlighted",
        "readwise:new-updated",
    ]


def test_readwise_respects_entity_types(tmp_path):
    export = tmp_path / "readwise.json"
    export.write_text(json.dumps([{"id": "h", "text": "Highlight"}]), encoding="utf-8")

    result = ReadwiseAdapter(path=str(export)).ingest(entity_types=["saved_item"])

    assert result.units == []
    assert result.edges == []


def test_readwise_adapter_is_registered():
    assert "readwise" in list_adapters()
    adapter = get_adapter("readwise", path="/tmp/readwise.json")
    assert isinstance(adapter, ReadwiseAdapter)
    assert adapter.name == "readwise"
