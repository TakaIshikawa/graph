from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.apple_books_highlights_json import AppleBooksHighlightsJsonAdapter
from graph.types.enums import EdgeRelation, EdgeSource
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
            "highlightColor": "Yellow",
            "location": "p. 42",
            "created": "2026-05-01T09:00:00Z"
          },
          {
            "id": "n1",
            "bookTitle": "Designing Data",
            "note": "Connect this to observability.",
            "annotationStyle": "Underline",
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
    assert highlight.metadata["color"] == "Yellow"
    assert highlight.metadata["normalized_color"] == "yellow"
    assert "Highlight:\nSystems need feedback." in highlight.content
    assert note.tags == ["apple_books", "note"]
    assert note.metadata["style"] == "Underline"
    assert note.metadata["normalized_color"] == "underline"
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


def test_apple_books_highlights_json_author_aggregates_and_edges(tmp_path):
    export = tmp_path / "books.json"
    export.write_text(
        """[
          {"id": "h1", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "Ada Reader", "selectedText": "Systems need feedback.", "location": "p. 42", "created": "2026-05-01T09:00:00Z"},
          {"id": "h2", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "ada reader", "selectedText": "Feedback loops matter.", "location": "p. 44", "created": "2026-05-01T10:00:00Z"},
          {"id": "n1", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "Ada Reader", "note": "Connect this.", "location": "p. 43", "modified": "2026-05-02T09:00:00Z"}
        ]""",
        encoding="utf-8",
    )

    authors = AppleBooksHighlightsJsonAdapter(path=str(export)).ingest(entity_types=["author"])
    with_edges = AppleBooksHighlightsJsonAdapter(path=str(export)).ingest(entity_types=["book", "highlight", "author"])

    assert "author" in AppleBooksHighlightsJsonAdapter(path=str(export)).entity_types
    author = authors.units[0]
    assert author.source_entity_type == "author"
    assert author.title == "Ada Reader"
    assert author.metadata["book_count"] == 1
    assert author.metadata["highlight_count"] == 2
    assert len(author.metadata["book_source_ids"]) == 1
    assert len(author.metadata["highlight_source_ids"]) == 2
    assert author.metadata["linked_source_ids"] == sorted(
        author.metadata["book_source_ids"] + author.metadata["highlight_source_ids"]
    )
    assert authors.edges == []

    author_edges = [edge for edge in with_edges.edges if edge.metadata["relation_type"] in {"book_author", "highlight_author"}]
    assert len(author_edges) == 3
    assert {edge.relation for edge in author_edges} == {EdgeRelation.RELATES_TO}
    assert {edge.source for edge in author_edges} == {EdgeSource.SOURCE}
    assert {edge.metadata["relation_type"] for edge in author_edges} == {"book_author", "highlight_author"}


def test_apple_books_highlights_json_highlight_color_aggregates_and_edges(tmp_path):
    export = tmp_path / "books.json"
    export.write_text(
        """[
          {"id": "h1", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "Ada Reader", "selectedText": "Systems need feedback.", "location": "p. 42", "color": "Yellow", "created": "2026-05-01T09:00:00Z"},
          {"id": "h2", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "Ada Reader", "selectedText": "Feedback loops matter.", "location": "p. 44", "highlightColor": "yellow", "annotationStyle": "Highlight", "created": "2026-05-01T10:00:00Z"},
          {"id": "h3", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "Ada Reader", "selectedText": "Different color.", "location": "p. 45", "style": "Blue", "created": "2026-05-01T11:00:00Z"},
          {"id": "n1", "assetId": "asset-1", "bookTitle": "Designing Data", "author": "Ada Reader", "note": "Connect this.", "location": "p. 43", "color": "Yellow", "modified": "2026-05-02T09:00:00Z"}
        ]""",
        encoding="utf-8",
    )

    colors = AppleBooksHighlightsJsonAdapter(path=str(export)).ingest(entity_types=["highlight_color"])
    with_edges = AppleBooksHighlightsJsonAdapter(path=str(export)).ingest(entity_types=["highlight_color", "highlight"])

    assert "highlight_color" in AppleBooksHighlightsJsonAdapter(path=str(export)).entity_types
    assert [unit.source_entity_type for unit in colors.units] == ["highlight_color", "highlight_color"]
    yellow = next(unit for unit in colors.units if unit.metadata["normalized_color"] == "yellow")
    assert yellow.metadata["book_title"] == "Designing Data"
    assert yellow.metadata["color"] == "Yellow"
    assert yellow.metadata["highlight_count"] == 2
    assert yellow.metadata["locations"] == ["p. 42", "p. 44"]
    assert yellow.metadata["selected_text_snippets"] == ["Systems need feedback.", "Feedback loops matter."]
    assert len(yellow.metadata["annotation_source_ids"]) == 2
    assert colors.edges == []

    color_edges = [edge for edge in with_edges.edges if edge.metadata["relation_type"] == "highlight_color_highlight"]
    assert len(color_edges) == 3
    assert {edge.relation for edge in color_edges} == {EdgeRelation.RELATES_TO}
    assert {edge.source for edge in color_edges} == {EdgeSource.SOURCE}
    assert {edge.from_unit_id for edge in color_edges if edge.to_unit_id in yellow.metadata["annotation_source_ids"]} == {yellow.source_id}
