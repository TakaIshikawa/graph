from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.apple_books_highlights_csv import AppleBooksHighlightsCsvAdapter
from graph.types.enums import ContentType


def test_apple_books_highlights_csv_ingests_representative_row(tmp_path):
    export = tmp_path / "apple_books.csv"
    export.write_text(
        "Book Title,Author,Highlight,Note,Chapter,Page,Location,Color,Created Date,Source Identifier\n"
        "A Test Book,Jane Writer,Highlighted passage,My note,Chapter 1,12,loc-120,Blue,2025-01-02T03:04:05Z,asset-123\n",
        encoding="utf-8",
    )

    result = AppleBooksHighlightsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "apple_books_highlights_csv"
    assert unit.source_id == "apple_books_highlights_csv:asset-123"
    assert unit.source_entity_type == "highlight"
    assert unit.title == "Apple Books highlight: A Test Book"
    assert unit.content_type == ContentType.INSIGHT
    assert "Book: A Test Book" in unit.content
    assert "Highlight: Highlighted passage" in unit.content
    assert "Note: My note" in unit.content
    assert unit.metadata["book_title"] == "A Test Book"
    assert unit.metadata["author"] == "Jane Writer"
    assert unit.metadata["highlight"] == "Highlighted passage"
    assert unit.metadata["note"] == "My note"
    assert unit.metadata["chapter"] == "Chapter 1"
    assert unit.metadata["page"] == "12"
    assert unit.metadata["location"] == "loc-120"
    assert unit.metadata["color"] == "Blue"
    assert unit.metadata["created_date"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["source_identifier"] == "asset-123"
    assert unit.tags == ["apple-books", "highlight", "blue"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_apple_books_highlights_csv_handles_alternate_headers_and_stable_ids(tmp_path):
    export = tmp_path / "alternate.csv"
    export.write_text(
        "Title,Authors,Selected Text,Comment,Section,Location in Book,Highlight Color,Date\n"
        "Alternate Book,Jane Writer,Alternate highlight,Alternate note,Intro,cfi-77,Yellow,2025-02-03\n",
        encoding="utf-8",
    )

    first = AppleBooksHighlightsCsvAdapter(path=str(export)).ingest().units[0]
    second = AppleBooksHighlightsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("apple_books_highlights_csv:")
    assert first.metadata["book_title"] == "Alternate Book"
    assert first.metadata["author"] == "Jane Writer"
    assert first.metadata["highlight"] == "Alternate highlight"
    assert first.metadata["note"] == "Alternate note"
    assert first.metadata["chapter"] == "Intro"
    assert first.metadata["location"] == "cfi-77"
    assert first.metadata["color"] == "Yellow"
    assert first.tags == ["apple-books", "highlight", "yellow"]
