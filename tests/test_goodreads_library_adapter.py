from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.goodreads_library import GoodreadsLibraryAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_goodreads_library_imports_shelves_and_optional_fields(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Book Id",
                "Title",
                "Author",
                "ISBN",
                "ISBN13",
                "My Rating",
                "Exclusive Shelf",
                "Bookshelves",
                "Date Read",
                "Date Added",
                "My Review",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Book Id": "1",
                "Title": "Read Book",
                "Author": "Ada",
                "ISBN": '="1234567890"',
                "ISBN13": '="9781234567890"',
                "My Rating": "5",
                "Exclusive Shelf": "read",
                "Bookshelves": "favorites, ai",
                "Date Read": "2025/01/03",
                "Date Added": "2025/01/01",
                "My Review": "Useful notes",
            }
        )
        writer.writerow(
            {
                "Book Id": "2",
                "Title": "Current Book",
                "Author": "Grace",
                "My Rating": "",
                "Exclusive Shelf": "currently-reading",
                "Bookshelves": "",
                "Date Added": "2025/01/02",
                "My Review": "",
            }
        )
        writer.writerow(
            {
                "Book Id": "3",
                "Title": "Future Book",
                "Author": "Katherine",
                "Exclusive Shelf": "to-read",
                "Date Added": "2025/01/04",
            }
        )

    result = GoodreadsLibraryAdapter(path=str(path)).ingest()

    assert len(result.units) == 3
    read = next(unit for unit in result.units if unit.metadata["book_id"] == "1")
    assert read.source_project == SourceProject.GOODREADS_LIBRARY
    assert read.source_id == "goodreads_library:1"
    assert read.title == "Read Book by Ada"
    assert read.metadata["isbn13"] == "9781234567890"
    assert read.metadata["rating"] == 5
    assert read.metadata["review"] == "Useful notes"
    assert read.tags == ["read", "favorites", "ai"]
    assert read.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert read.updated_at == datetime(2025, 1, 3, tzinfo=timezone.utc)
    current = next(unit for unit in result.units if unit.metadata["book_id"] == "2")
    assert current.metadata["rating"] is None
    assert current.metadata["review"] == ""
    assert current.tags == ["currently-reading"]
    future = next(unit for unit in result.units if unit.metadata["book_id"] == "3")
    assert future.tags == ["to-read"]


def test_goodreads_library_adapter_is_registered():
    assert "goodreads_library" in list_adapters()
    assert get_adapter("goodreads_library", path="/tmp/books.csv").name == "goodreads_library"
