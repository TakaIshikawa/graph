from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.omnivore_json import OmnivoreJsonAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import SyncState


def test_omnivore_json_ingests_articles_highlights_and_notes(tmp_path):
    export = tmp_path / "omnivore.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "page-1",
                    "title": "Research Article",
                    "url": "https://example.com/research",
                    "author": "Ada Lovelace",
                    "state": "ARCHIVED",
                    "labels": [{"name": "Research"}, {"name": "AI"}],
                    "savedAt": "2025-02-01T10:00:00Z",
                    "readAt": "2025-02-02T12:00:00Z",
                    "highlights": [
                        {
                            "id": "hl-1",
                            "quote": "A useful passage",
                            "annotation": "Connect to project notes",
                            "color": "yellow",
                            "createdAt": "2025-02-02T11:00:00Z",
                        },
                        {
                            "id": "note-1",
                            "annotation": "Article-level note",
                            "createdAt": "2025-02-02T11:30:00Z",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = OmnivoreJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 3
    article = next(unit for unit in result.units if unit.source_entity_type == "article")
    assert article.source_project == SourceProject.OMNIVORE_JSON
    assert article.source_id == "omnivore:page-1"
    assert article.title == "Research Article"
    assert article.content_type == ContentType.ARTIFACT
    assert article.metadata["url"] == "https://example.com/research"
    assert article.metadata["author"] == "Ada Lovelace"
    assert article.metadata["state"] == "ARCHIVED"
    assert article.metadata["labels"] == ["research", "ai"]
    assert article.metadata["saved_at"] == "2025-02-01T10:00:00Z"
    assert article.metadata["read_at"] == "2025-02-02T12:00:00Z"
    assert article.metadata["archived"] is True
    assert article.metadata["read"] is True
    assert article.tags == ["research", "ai"]

    highlight = next(unit for unit in result.units if unit.source_entity_type == "highlight")
    assert highlight.source_id == "omnivore_highlight:hl-1"
    assert highlight.content_type == ContentType.INSIGHT
    assert "A useful passage" in highlight.content
    assert "Note: Connect to project notes" in highlight.content
    assert highlight.metadata["article_source_id"] == "omnivore:page-1"
    assert highlight.metadata["color"] == "yellow"

    note = next(unit for unit in result.units if unit.source_entity_type == "note")
    assert note.source_id == "omnivore_highlight:note-1"
    assert note.metadata["note"] == "Article-level note"

    assert len(result.edges) == 2
    assert {edge.from_unit_id for edge in result.edges} == {"omnivore:page-1"}
    assert {edge.to_unit_id for edge in result.edges} == {
        "omnivore_highlight:hl-1",
        "omnivore_highlight:note-1",
    }
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}


def test_omnivore_json_accepts_nested_single_object_shape(tmp_path):
    export = tmp_path / "omnivore.json"
    export.write_text(
        json.dumps(
            {
                "page": {
                    "id": "nested-page",
                    "title": "Nested Page",
                    "url": "https://example.com/nested",
                    "authorName": "Grace Hopper",
                    "isRead": True,
                    "labels": ["Programming"],
                },
                "highlights": {
                    "a": {
                        "text": "Nested highlight",
                        "note": "Nested note",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    result = OmnivoreJsonAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["article", "highlight"]
    article = result.units[0]
    assert article.source_id == "omnivore:nested-page"
    assert article.metadata["author"] == "Grace Hopper"
    assert article.metadata["read"] is True
    assert article.tags == ["programming"]
    assert result.units[1].source_id.startswith("omnivore_highlight:")
    assert result.edges[0].from_unit_id == "omnivore:nested-page"


def test_omnivore_json_uses_url_source_id_without_omnivore_id(tmp_path):
    export = tmp_path / "omnivore.json"
    export.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "title": "URL Only",
                        "url": "https://example.com/url-only",
                        "labels": "one, two",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = OmnivoreJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == "url:https://example.com/url-only"
    assert result.units[0].metadata["labels"] == ["one", "two"]


def test_omnivore_json_entity_type_filtering(tmp_path):
    export = tmp_path / "omnivore.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "page-2",
                    "title": "Filtered",
                    "url": "https://example.com/filtered",
                    "highlights": [{"id": "hl-2", "quote": "Keep only highlight"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = OmnivoreJsonAdapter(path=str(export)).ingest(entity_types=["highlight"])

    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "highlight"
    assert result.edges[0].from_unit_id == "omnivore:page-2"


def test_omnivore_json_filters_by_sync_state(tmp_path):
    export = tmp_path / "omnivore.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "old",
                    "title": "Old",
                    "url": "https://example.com/old",
                    "updatedAt": "2025-01-01T00:00:00Z",
                },
                {
                    "id": "new",
                    "title": "New",
                    "url": "https://example.com/new",
                    "updatedAt": "2025-01-03T00:00:00Z",
                },
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="omnivore_json",
        source_entity_type="article",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = OmnivoreJsonAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert result.units[0].source_id == "omnivore:new"


def test_omnivore_json_handles_bad_input(tmp_path):
    export = tmp_path / "omnivore.json"
    export.write_text("not json", encoding="utf-8")

    result = OmnivoreJsonAdapter(path=str(export)).ingest()

    assert result.units == []
    assert result.edges == []
    assert OmnivoreJsonAdapter(path="/does/not/exist.json").ingest().units == []


def test_omnivore_json_adapter_is_registered():
    assert "omnivore_json" in list_adapters()
    adapter = get_adapter("omnivore_json", path="/tmp/omnivore.json")
    assert adapter.name == "omnivore_json"
