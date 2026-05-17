from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.adapters.goodreads_library_csv import GoodreadsLibraryCsvAdapter
from graph.types.enums import ContentType


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_goodreads_library_csv_ingests_representative_library_row(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    _write_csv(
        path,
        [
            {
                "Book Id": "12345",
                "Title": "The Left Hand of Darkness",
                "Author": "Ursula K. Le Guin",
                "Author l-f": "Le Guin, Ursula K.",
                "ISBN": '="0441478123"',
                "ISBN13": '="9780441478125"',
                "My Rating": "5",
                "Average Rating": "4.11",
                "Publisher": "Ace",
                "Date Read": "2025/01/15",
                "Date Added": "2025/01/10",
                "Bookshelves": "science-fiction, favorites",
                "Exclusive Shelf": "read",
                "My Review": "Cold, precise, and humane.",
            }
        ],
    )

    result = GoodreadsLibraryCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "goodreads_library_csv"
    assert unit.source_id == "goodreads_library_csv:12345"
    assert unit.source_entity_type == "book"
    assert unit.title == "The Left Hand of Darkness by Ursula K. Le Guin"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Title: The Left Hand of Darkness" in unit.content
    assert "Author: Ursula K. Le Guin" in unit.content
    assert "ISBN: 0441478123" in unit.content
    assert "ISBN13: 9780441478125" in unit.content
    assert "My rating: 5/5" in unit.content
    assert "Average rating: 4.11/5" in unit.content
    assert "Publisher: Ace" in unit.content
    assert "Shelves: science-fiction, favorites, read" in unit.content
    assert "Cold, precise, and humane." in unit.content
    assert unit.metadata["book_id"] == "12345"
    assert unit.metadata["isbn"] == "0441478123"
    assert unit.metadata["isbn13"] == "9780441478125"
    assert unit.metadata["rating"] == 5
    assert unit.metadata["my_rating"] == 5
    assert unit.metadata["average_rating"] == 4.11
    assert unit.metadata["publisher"] == "Ace"
    assert unit.metadata["date_read"] == "2025-01-15T00:00:00+00:00"
    assert unit.metadata["date_added"] == "2025-01-10T00:00:00+00:00"
    assert unit.metadata["shelves"] == ["science-fiction", "favorites"]
    assert unit.metadata["exclusive_shelf"] == "read"
    assert unit.metadata["review"] == "Cold, precise, and humane."
    assert unit.tags == ["goodreads", "science-fiction", "favorites", "read"]
    assert unit.created_at == datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 15, tzinfo=timezone.utc)


def test_goodreads_library_csv_handles_missing_optional_columns_from_file_like():
    handle = StringIO("Title,Author,Date Added\nMinimal Book,Jane Doe,2025-02-03\n")

    result = GoodreadsLibraryCsvAdapter(path=handle).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Minimal Book by Jane Doe"
    assert unit.metadata["title"] == "Minimal Book"
    assert unit.metadata["author"] == "Jane Doe"
    assert "isbn" not in unit.metadata
    assert "rating" not in unit.metadata
    assert "review" not in unit.metadata
    assert unit.metadata["date_added"] == "2025-02-03T00:00:00+00:00"
    assert unit.tags == ["goodreads"]


def test_goodreads_library_csv_source_ids_are_deterministic_without_book_id(tmp_path):
    path = tmp_path / "goodreads.csv"
    _write_csv(
        path,
        [
            {
                "Title": "Stable Book",
                "Author": "Ada Lovelace",
                "ISBN13": "9780000000002",
                "Date Added": "2025-01-01",
            }
        ],
    )

    first = GoodreadsLibraryCsvAdapter(path=str(path)).ingest().units[0]
    second = GoodreadsLibraryCsvAdapter(path=str(path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("goodreads_library_csv:")
    assert first.source_id != "goodreads_library_csv:"
