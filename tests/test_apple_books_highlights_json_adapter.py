from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.apple_books_highlights_json import AppleBooksHighlightsJsonAdapter
from graph.types.models import SyncState


def test_apple_books_highlights_json_ingests_highlights_and_notes(tmp_path):
    export = tmp_path / "books.json"
    export.write_text(
        """[
          {
            "id": "h1",
            "bookTitle": "Designing Data",
            "author": "Ada Reader",
            "selectedText": "Systems need feedback.",
            "location": "p. 42",
            "created": "2026-05-01T09:00:00Z"
          },
          {
            "id": "n1",
            "bookTitle": "Designing Data",
            "note": "Connect this to observability.",
            "location": "p. 43",
            "modified": "2026-05-02T09:00:00Z"
          }
        ]""",
        encoding="utf-8",
    )

    result = AppleBooksHighlightsJsonAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["highlight", "note"]
    highlight = result.units[0]
    note = result.units[1]
    assert highlight.metadata["book_title"] == "Designing Data"
    assert highlight.metadata["author"] == "Ada Reader"
    assert highlight.metadata["location"] == "p. 42"
    assert "Highlight:\nSystems need feedback." in highlight.content
    assert note.tags == ["apple_books", "note"]
    assert "Note:\nConnect this to observability." in note.content


def test_apple_books_highlights_json_filters_since_by_modified_or_created(tmp_path):
    export = tmp_path / "books.json"
    export.write_text(
        """{"annotations": [
          {"bookTitle": "Old", "selectedText": "old", "created": "2026-05-01T00:00:00Z"},
          {"bookTitle": "New", "selectedText": "new", "created": "2026-05-01T00:00:00Z", "modified": "2026-05-03T00:00:00Z"}
        ]}""",
        encoding="utf-8",
    )
    since = SyncState(source_project="apple_books_highlights_json", source_entity_type="highlight", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = AppleBooksHighlightsJsonAdapter(path=str(export)).ingest(since=since)
    notes = AppleBooksHighlightsJsonAdapter(path=str(export)).ingest(entity_types=["note"])

    assert [unit.metadata["book_title"] for unit in result.units] == ["New"]
    assert notes.units == []


def test_apple_books_highlights_json_book_aggregates_and_edges(tmp_path):
    export = tmp_path / "books.json"
    export.write_text(
        """[
          {"id": "h1", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "Ada Reader", "selectedText": "Systems need feedback.", "location": "p. 42", "created": "2026-05-01T09:00:00Z"},
          {"id": "n1", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "Ada Reader", "note": "Connect this.", "location": "p. 43", "modified": "2026-05-02T09:00:00Z"}
        ]""",
        encoding="utf-8",
    )

    result = AppleBooksHighlightsJsonAdapter(path=str(export)).ingest(entity_types=["book", "highlight", "note"])
    book = next(unit for unit in result.units if unit.source_entity_type == "book")

    assert book.metadata["annotation_count"] == 2
    assert book.metadata["highlight_count"] == 1
    assert book.metadata["note_count"] == 1
    assert book.metadata["locations"] == ["p. 42", "p. 43"]
    assert {edge.metadata["relation_type"] for edge in result.edges} == {"book_contains_highlight", "book_contains_note"}
