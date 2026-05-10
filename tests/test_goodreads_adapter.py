from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.goodreads import GoodreadsAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_goodreads_csv_ingests_books_with_full_metadata(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Book Id,Title,Author,ISBN,ISBN13,My Rating,Average Rating,Publisher,Year Published,Date Read,Date Added,Bookshelves,Exclusive Shelf,My Review,Read Count",
                '12345,The Great Book,Jane Doe,1234567890,9781234567890,5,4.2,Publisher Co,2020,2025/01/15,2025/01/10,"to-read,favorites",read,"Amazing book, highly recommend!",1',
            ]
        ),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.GOODREADS
    assert unit.source_id == "goodreads:12345"
    assert unit.source_entity_type == "book"
    assert unit.title == "The Great Book by Jane Doe"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Title: The Great Book" in unit.content
    assert "Author: Jane Doe" in unit.content
    assert "Rating: 5/5" in unit.content
    assert "Shelves: to-read, favorites" in unit.content
    assert "Amazing book, highly recommend!" in unit.content
    assert unit.metadata == {
        "book_id": "12345",
        "title": "The Great Book",
        "author": "Jane Doe",
        "isbn": "9781234567890",
        "my_rating": "5",
        "average_rating": "4.2",
        "publisher": "Publisher Co",
        "year_published": "2020",
        "date_read": "2025/01/15",
        "date_added": "2025/01/10",
        "shelves": ["to-read", "favorites"],
        "exclusive_shelf": "read",
        "review": "Amazing book, highly recommend!",
        "read_count": "1",
    }
    assert unit.tags == ["to-read", "favorites"]
    assert unit.created_at == datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 15, tzinfo=timezone.utc)


def test_goodreads_csv_handles_missing_optional_fields(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Book Id,Title,Author,Date Added",
                "67890,Minimal Book,John Smith,2025/01/20",
            ]
        ),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "goodreads:67890"
    assert unit.title == "Minimal Book by John Smith"
    assert unit.metadata["title"] == "Minimal Book"
    assert unit.metadata["author"] == "John Smith"
    assert unit.metadata["isbn"] == ""
    assert unit.metadata["my_rating"] == ""
    assert unit.metadata["review"] == ""
    assert unit.metadata["shelves"] == []
    assert unit.tags == []
    assert unit.created_at == datetime(2025, 1, 20, tzinfo=timezone.utc)


def test_goodreads_csv_uses_isbn_for_source_id_when_book_id_missing(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Title,Author,ISBN13",
                "Book Without ID,Author Name,9780123456789",
            ]
        ),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "isbn:9780123456789"


def test_goodreads_csv_generates_hash_source_id_when_no_book_id_or_isbn(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Title,Author",
                "Unknown Book,Unknown Author",
            ]
        ),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id.startswith("goodreads:")
    assert len(unit.source_id) == 34  # "goodreads:" + 24 character hash


def test_goodreads_json_ingests_books_from_list_or_dict_exports(tmp_path):
    export = tmp_path / "goodreads.json"
    export.write_text(
        json.dumps(
            {
                "books": [
                    {
                        "book_id": "abc123",
                        "title": "JSON Book",
                        "author": "JSON Author",
                        "isbn13": "9781111111111",
                        "my_rating": "4",
                        "date_read": "2025-01-25T10:00:00Z",
                        "bookshelves": "currently-reading",
                        "review": "Great read!",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    unit = GoodreadsAdapter(path=str(export)).ingest().units[0]

    assert unit.source_id == "goodreads:abc123"
    assert unit.title == "JSON Book by JSON Author"
    assert unit.metadata["title"] == "JSON Book"
    assert unit.metadata["author"] == "JSON Author"
    assert unit.metadata["isbn"] == "9781111111111"
    assert unit.metadata["my_rating"] == "4"
    assert unit.metadata["review"] == "Great read!"
    assert unit.metadata["shelves"] == ["currently-reading"]
    assert unit.tags == ["currently-reading"]
    assert unit.created_at == datetime(2025, 1, 25, 10, 0, tzinfo=timezone.utc)


def test_goodreads_since_filters_books_not_newer_than_sync_state(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Book Id,Title,Author,Date Read",
                "1,Old Book,Author 1,2025/01/01",
                "2,Equal Book,Author 2,2025/01/02",
                "3,New Book,Author 3,2025/01/03",
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="goodreads",
        source_entity_type="book",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = GoodreadsAdapter(path=str(export)).ingest(since=since)

    assert [unit.source_id for unit in result.units] == ["goodreads:3"]


def test_goodreads_respects_entity_types(tmp_path):
    export = tmp_path / "goodreads.json"
    export.write_text(
        json.dumps([{"book_id": "123", "title": "Test Book", "author": "Test Author"}]),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest(entity_types=["article"])

    assert result.units == []
    assert result.edges == []


def test_goodreads_skips_books_without_title_and_author(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Book Id,Title,Author",
                "1,,,",  # Empty title and author
                "2,Valid Book,Valid Author",
            ]
        ),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == "goodreads:2"


def test_goodreads_handles_malformed_csv(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text("This is not a valid CSV file\n<<>><>", encoding="utf-8")

    result = GoodreadsAdapter(path=str(export)).ingest()

    # Should return empty result without crashing
    assert result.units == []


def test_goodreads_handles_empty_export(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Book Id,Title,Author",
                # No data rows
            ]
        ),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest()

    assert result.units == []


def test_goodreads_handles_nonexistent_file():
    result = GoodreadsAdapter(path="/nonexistent/path.csv").ingest()

    assert result.units == []


def test_goodreads_parses_shelves_correctly(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Book Id,Title,Author,Bookshelves",
                '1,Book 1,Author 1,"fiction,science-fiction,favorites"',
                "2,Book 2,Author 2,non-fiction",
                "3,Book 3,Author 3,",  # Empty shelves
            ]
        ),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest()

    assert len(result.units) == 3
    assert result.units[0].tags == ["fiction", "science-fiction", "favorites"]
    assert result.units[1].tags == ["non-fiction"]
    assert result.units[2].tags == []


def test_goodreads_uses_date_added_when_date_read_missing(tmp_path):
    export = tmp_path / "goodreads.csv"
    export.write_text(
        "\n".join(
            [
                "Book Id,Title,Author,Date Added",
                "1,Book 1,Author 1,2025/01/10",
            ]
        ),
        encoding="utf-8",
    )

    result = GoodreadsAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.created_at == datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 10, tzinfo=timezone.utc)


def test_goodreads_adapter_is_registered():
    assert "goodreads" in list_adapters()
    adapter = get_adapter("goodreads", path="/tmp/goodreads.csv")
    assert adapter.name == "goodreads"
