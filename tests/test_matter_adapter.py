from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.matter import MatterAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_matter_json_ingests_articles_with_highlights(tmp_path):
    export = tmp_path / "matter.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "123",
                    "title": "Article Title",
                    "url": "https://example.com/article",
                    "author": "John Doe",
                    "content": "<p>Article content here</p>",
                    "tags": ["research", "important"],
                    "reading_progress": "75",
                    "created_at": "2025-01-10T12:00:00Z",
                    "updated_at": "2025-01-11T14:30:00Z",
                    "highlights": [
                        {
                            "id": "h1",
                            "text": "Important quote",
                            "note": "My note",
                            "created_at": "2025-01-11T10:00:00Z",
                        },
                        {
                            "id": "h2",
                            "text": "Another highlight",
                            "created_at": "2025-01-11T11:00:00Z",
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = MatterAdapter(path=str(export)).ingest()

    # Should create 1 article + 2 highlight units
    assert len(result.units) == 3

    # Check article unit
    article = [u for u in result.units if u.source_entity_type == "article"][0]
    assert article.source_project == SourceProject.MATTER
    assert article.source_id == "matter:123"
    assert article.title == "Article Title"
    assert article.content_type == ContentType.ARTIFACT
    assert "Article Title" in article.content
    assert "Author: John Doe" in article.content
    assert "URL: https://example.com/article" in article.content
    assert "Highlights (2)" in article.content
    assert article.metadata["reading_progress"] == "75"
    assert article.metadata["highlight_count"] == "2"
    assert article.tags == ["research", "important"]

    # Check highlight units
    highlights = [u for u in result.units if u.source_entity_type == "highlight"]
    assert len(highlights) == 2
    assert highlights[0].source_id == "matter_highlight:h1"
    assert "Highlight: Important quote" in highlights[0].content
    assert "Note: My note" in highlights[0].content
    assert highlights[1].source_id == "matter_highlight:h2"
    assert "Highlight: Another highlight" in highlights[1].content


def test_matter_json_ingests_articles_without_highlights(tmp_path):
    export = tmp_path / "matter.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "456",
                    "title": "Simple Article",
                    "url": "https://example.com/simple",
                    "created_at": "2025-01-12T10:00:00Z",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = MatterAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_entity_type == "article"
    assert unit.source_id == "matter:456"
    assert unit.metadata["highlight_count"] == ""


def test_matter_entity_type_filtering_articles_only(tmp_path):
    export = tmp_path / "matter.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "789",
                    "url": "https://example.com/test",
                    "title": "Test",
                    "highlights": [{"id": "h1", "text": "Highlight text"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = MatterAdapter(path=str(export)).ingest(entity_types=["article"])

    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "article"


def test_matter_entity_type_filtering_highlights_only(tmp_path):
    export = tmp_path / "matter.json"
    export.write_text(
        json.dumps(
            [
                {
                    "id": "789",
                    "url": "https://example.com/test",
                    "title": "Test",
                    "highlights": [{"id": "h1", "text": "Highlight text"}],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = MatterAdapter(path=str(export)).ingest(entity_types=["highlight"])

    assert len(result.units) == 1
    assert result.units[0].source_entity_type == "highlight"


def test_matter_handles_nested_articles(tmp_path):
    export = tmp_path / "matter.json"
    export.write_text(
        json.dumps(
            {
                "articles": [
                    {
                        "id": "nested",
                        "title": "Nested Article",
                        "url": "https://example.com/nested",
                        "created_at": "2025-01-13T10:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MatterAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == "matter:nested"


def test_matter_filters_by_sync_state(tmp_path):
    export = tmp_path / "matter.json"
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
        source_project="matter",
        source_entity_type="article",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = MatterAdapter(path=str(export)).ingest(since=since)

    assert len(result.units) == 1
    assert result.units[0].source_id == "matter:new"


def test_matter_respects_entity_types(tmp_path):
    export = tmp_path / "matter.json"
    export.write_text(
        json.dumps([{"url": "https://example.com", "title": "Test"}]), encoding="utf-8"
    )

    result = MatterAdapter(path=str(export)).ingest(entity_types=["bookmark"])

    assert result.units == []


def test_matter_handles_malformed_json(tmp_path):
    export = tmp_path / "matter.json"
    export.write_text("not valid json", encoding="utf-8")

    result = MatterAdapter(path=str(export)).ingest()

    assert result.units == []
    assert result.edges == []


def test_matter_handles_missing_file():
    result = MatterAdapter(path="/nonexistent/file.json").ingest()
    assert result.units == []
    assert result.edges == []


def test_matter_adapter_is_registered():
    assert "matter" in list_adapters()
    adapter = get_adapter("matter", path="/tmp/matter.json")
    assert adapter.name == "matter"
