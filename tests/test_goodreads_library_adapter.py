from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.goodreads_library import GoodreadsLibraryAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import EdgeRelation, SourceProject


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

    assert len([unit for unit in result.units if unit.source_entity_type == "book"]) == 3
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
    current = next(unit for unit in result.units if unit.source_entity_type == "book" and unit.metadata["book_id"] == "2")
    assert current.metadata["rating"] is None
    assert current.metadata["review"] == ""
    assert current.tags == ["currently-reading"]
    future = next(unit for unit in result.units if unit.source_entity_type == "book" and unit.metadata["book_id"] == "3")
    assert future.tags == ["to-read"]

    authors = [unit for unit in result.units if unit.source_entity_type == "author"]
    shelves = [unit for unit in result.units if unit.source_entity_type == "shelf"]
    assert {unit.metadata["author"] for unit in authors} == {"Ada", "Grace", "Katherine"}
    assert {unit.metadata["shelf"] for unit in shelves} == {"read", "favorites", "ai", "currently-reading", "to-read"}
    shelf_edge = next(edge for edge in result.edges if edge.relation == EdgeRelation.CONTAINS)
    assert shelf_edge.metadata["relation_type"] == "shelf_contains_book"
    author_edge = next(edge for edge in result.edges if edge.relation == EdgeRelation.RELATES_TO)
    assert author_edge.metadata["relation_type"] == "book_author"


def test_goodreads_library_entity_filters_for_author_shelf_and_book(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Book Id", "Title", "Author", "Exclusive Shelf", "Bookshelves"])
        writer.writeheader()
        writer.writerow({"Book Id": "1", "Title": "One", "Author": "Ada", "Exclusive Shelf": "read", "Bookshelves": "favorites"})
        writer.writerow({"Book Id": "2", "Title": "Two", "Author": "Ada", "Exclusive Shelf": "read", "Bookshelves": ""})

    author_only = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["author"])
    shelf_and_book = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["shelf", "book"])
    book_only = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["book"])

    assert [unit.source_entity_type for unit in author_only.units] == ["author"]
    assert author_only.units[0].metadata["book_count"] == 2
    assert author_only.edges == []
    assert {unit.source_entity_type for unit in shelf_and_book.units} == {"book", "shelf"}
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in shelf_and_book.edges)
    assert {unit.source_entity_type for unit in book_only.units} == {"book"}
    assert book_only.edges == []


def test_goodreads_library_ingests_series_from_columns_and_titles(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Book Id", "Title", "Author", "Series Name", "Series Number", "Date Added"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Book Id": "1",
                "Title": "First",
                "Author": "Ada",
                "Series Name": "Pattern Books",
                "Series Number": "1",
                "Date Added": "2025/01/01",
            }
        )
        writer.writerow(
            {
                "Book Id": "2",
                "Title": "Second (Pattern Books, #2)",
                "Author": "Ada",
                "Series Name": "",
                "Series Number": "",
                "Date Added": "2025/01/02",
            }
        )

    result = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["book", "series"])

    series_units = [unit for unit in result.units if unit.source_entity_type == "series"]
    assert len(series_units) == 1
    series = series_units[0]
    assert series.source_id.startswith("goodreads_library:series:")
    assert series.metadata["series"] == "Pattern Books"
    assert series.metadata["book_count"] == 2
    assert [book["sequence"] for book in series.metadata["books"]] == [1, 2]
    assert [book["source_id"] for book in series.metadata["books"]] == [
        "goodreads_library:1",
        "goodreads_library:2",
    ]
    second = next(unit for unit in result.units if unit.metadata.get("book_id") == "2")
    assert second.metadata["series"] == {"name": "Pattern Books", "source": "title", "sequence": 2}
    series_edges = [edge for edge in result.edges if edge.metadata["relation_type"] == "series_contains_book"]
    assert len(series_edges) == 2
    assert all(edge.from_unit_id == series.source_id for edge in series_edges)
    assert all(edge.relation == EdgeRelation.CONTAINS for edge in series_edges)


def test_goodreads_library_series_edges_follow_entity_filter(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Book Id", "Title", "Author", "Book Series"])
        writer.writeheader()
        writer.writerow({"Book Id": "1", "Title": "One", "Author": "Ada", "Book Series": "Solo"})

    series_only = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["series"])
    book_only = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["book"])

    assert [unit.source_entity_type for unit in series_only.units] == ["series"]
    assert series_only.edges == []
    assert [unit.source_entity_type for unit in book_only.units] == ["book"]
    assert book_only.units[0].metadata["series"]["name"] == "Solo"
    assert book_only.edges == []


def test_goodreads_library_ingests_publishers_and_edges_follow_filters(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Book Id", "Title", "Author", "Publisher", "Original Publisher"])
        writer.writeheader()
        writer.writerow({"Book Id": "1", "Title": "One", "Author": "Ada", "Publisher": "Tech Press", "Original Publisher": ""})
        writer.writerow({"Book Id": "2", "Title": "Two", "Author": "Grace", "Publisher": "tech press", "Original Publisher": ""})
        writer.writerow({"Book Id": "3", "Title": "Three", "Author": "Katherine", "Publisher": "", "Original Publisher": "Archive House"})

    result = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["book", "publisher"])

    publishers = [unit for unit in result.units if unit.source_entity_type == "publisher"]
    assert {unit.metadata["publisher"] for unit in publishers} == {"Tech Press", "Archive House"}
    tech = next(unit for unit in publishers if unit.metadata["publisher"] == "Tech Press")
    assert tech.source_id.startswith("goodreads_library:publisher:")
    assert tech.metadata["book_count"] == 2
    assert len([edge for edge in result.edges if edge.metadata["relation_type"] == "book_publisher"]) == 3

    publisher_only = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["publisher"])
    assert {unit.source_entity_type for unit in publisher_only.units} == {"publisher"}
    assert publisher_only.edges == []


def test_goodreads_library_ingests_owned_copy_units(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Book Id",
                "Title",
                "Author",
                "Condition",
                "Date Acquired",
                "Purchase Location",
                "Format",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Book Id": "1",
                "Title": "Owned Book",
                "Author": "Ada",
                "Condition": "Very Good",
                "Date Acquired": "2025/01/05",
                "Purchase Location": "Local Shop",
                "Format": "Hardcover",
            }
        )
        writer.writerow({"Book Id": "2", "Title": "Digital Book", "Author": "Grace"})

    result = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["book", "copy"])

    copies = [unit for unit in result.units if unit.source_entity_type == "copy"]
    assert len(copies) == 1
    copy = copies[0]
    book = next(unit for unit in result.units if unit.source_entity_type == "book" and unit.metadata["book_id"] == "1")
    assert copy.source_id.startswith("goodreads_library:copy:")
    assert copy.metadata["book_source_id"] == book.source_id
    assert copy.metadata["condition"] == "Very Good"
    assert copy.metadata["date_acquired"] == "2025/01/05"
    assert copy.metadata["purchase_location"] == "Local Shop"
    assert copy.metadata["format"] == "Hardcover"
    assert copy.created_at == datetime(2025, 1, 5, tzinfo=timezone.utc)
    copy_edges = [edge for edge in result.edges if edge.metadata["relation_type"] == "book_contains_copy"]
    assert len(copy_edges) == 1
    assert copy_edges[0].from_unit_id == book.source_id
    assert copy_edges[0].to_unit_id == copy.source_id


def test_goodreads_library_copy_edges_follow_entity_filters(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Book Id", "Title", "Author", "Condition"])
        writer.writeheader()
        writer.writerow({"Book Id": "1", "Title": "Owned Book", "Author": "Ada", "Condition": "Good"})

    copy_only = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["copy"])

    assert [unit.source_entity_type for unit in copy_only.units] == ["copy"]
    assert copy_only.edges == []


def test_goodreads_library_review_units_and_edges(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Book Id", "Title", "Author", "My Rating", "Date Read", "Date Added", "Bookshelves", "My Review"])
        writer.writeheader()
        writer.writerow(
            {
                "Book Id": "1",
                "Title": "Reviewed Book",
                "Author": "Ada",
                "My Rating": "4",
                "Date Read": "2025/01/03",
                "Date Added": "2025/01/01",
                "Bookshelves": "favorites",
                "My Review": "Useful review",
            }
        )
        writer.writerow({"Book Id": "2", "Title": "No Review", "Author": "Grace", "My Review": ""})

    result = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["book", "review"])
    review_only = GoodreadsLibraryAdapter(path=str(path)).ingest(entity_types=["review"])
    review = next(unit for unit in result.units if unit.source_entity_type == "review")

    assert review.metadata["review"] == "Useful review"
    assert review.metadata["rating"] == 4
    assert review.metadata["shelves"] == ["favorites"]
    assert len([unit for unit in result.units if unit.source_entity_type == "review"]) == 1
    assert [edge.metadata["relation_type"] for edge in result.edges] == ["book_contains_review"]
    assert [unit.source_entity_type for unit in review_only.units] == ["review"]
    assert review_only.edges == []


def test_goodreads_library_adapter_is_registered():
    assert "goodreads_library" in list_adapters()
    assert get_adapter("goodreads_library", path="/tmp/books.csv").name == "goodreads_library"
