from __future__ import annotations

from graph.adapters.apple_books_library_csv import AppleBooksLibraryCsvAdapter
from graph.adapters.registry import get_adapter


def test_apple_books_library_csv_ingests_books(tmp_path):
    path = tmp_path / "books.csv"
    path.write_text("Title,Author,Sort Author,Genre,Publisher,Publication Date,ISBN,Book ID,Reading Status,Percent Complete,Last Opened,Store URL\nExample Book,Ada Lovelace,\"Lovelace, Ada\",Technology,Press,2020-01-01,978123,b1,Reading,75,2026-05-01T12:00:00Z,https://books.test/b1\n", encoding="utf-8")

    unit = AppleBooksLibraryCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.source_project == "apple_books_library_csv"
    assert unit.source_id == "apple_books_library_csv:isbn:978123"
    assert unit.source_entity_type == "book"
    assert unit.metadata["author"] == "Ada Lovelace"
    assert unit.metadata["percent_complete"] == 75.0
    assert "Author: Ada Lovelace" in unit.content
    assert isinstance(get_adapter("apple-books-library-csv"), AppleBooksLibraryCsvAdapter)
