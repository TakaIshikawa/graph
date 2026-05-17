from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.adapters.goodreads_reviews_csv import GoodreadsReviewsCsvAdapter
from graph.types.enums import ContentType


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_goodreads_reviews_csv_ingests_representative_row(tmp_path):
    path = tmp_path / "goodreads_library_export.csv"
    _write_csv(
        path,
        [
            {
                "Book Id": "12345",
                "Title": "The Left Hand of Darkness",
                "Author": "Ursula K. Le Guin",
                "Author l-f": "Le Guin, Ursula K.",
                "Additional Authors": "Jane Editor; Sam Translator",
                "ISBN": '="0441478123"',
                "ISBN13": '="9780441478125"',
                "My Rating": "5",
                "Average Rating": "4.11",
                "Publisher": "Ace",
                "Binding": "Paperback",
                "Number of Pages": "304",
                "Year Published": "1987",
                "Original Publication Year": "1969",
                "Date Read": "2025/01/15",
                "Date Added": "2025/01/10",
                "Bookshelves": "science-fiction, favorites",
                "Exclusive Shelf": "read",
                "My Review": "Cold, precise, and humane.",
                "Spoiler": "false",
                "Private Notes": "Revisit the politics.",
                "Read Count": "2",
                "Owned Copies": "1",
            }
        ],
    )

    result = GoodreadsReviewsCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "goodreads_reviews_csv"
    assert unit.source_id == "goodreads_reviews_csv:12345"
    assert unit.source_entity_type == "book_review"
    assert unit.title == "The Left Hand of Darkness by Ursula K. Le Guin"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Title: The Left Hand of Darkness" in unit.content
    assert "Author: Ursula K. Le Guin" in unit.content
    assert "My rating: 5/5" in unit.content
    assert "Cold, precise, and humane." in unit.content
    assert unit.metadata["book_id"] == "12345"
    assert unit.metadata["isbn"] == "0441478123"
    assert unit.metadata["isbn13"] == "9780441478125"
    assert unit.metadata["my_rating"] == 5
    assert unit.metadata["average_rating"] == 4.11
    assert unit.metadata["publisher"] == "Ace"
    assert unit.metadata["binding"] == "Paperback"
    assert unit.metadata["page_count"] == 304
    assert unit.metadata["year_published"] == 1987
    assert unit.metadata["original_publication_year"] == 1969
    assert unit.metadata["date_read"] == "2025-01-15T00:00:00+00:00"
    assert unit.metadata["date_added"] == "2025-01-10T00:00:00+00:00"
    assert unit.metadata["shelves"] == ["science-fiction", "favorites"]
    assert unit.metadata["exclusive_shelf"] == "read"
    assert unit.metadata["spoiler"] is False
    assert unit.metadata["private_notes"] == "Revisit the politics."
    assert unit.metadata["read_count"] == 2
    assert unit.metadata["owned_copies"] == 1
    assert unit.tags == ["goodreads", "science-fiction", "favorites", "read"]
    assert unit.created_at == datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 15, tzinfo=timezone.utc)


def test_goodreads_reviews_csv_tolerates_missing_optional_fields_from_file_like():
    handle = StringIO("Title,Author,Date Added\nMinimal Book,Jane Doe,2025-02-03\n")

    result = GoodreadsReviewsCsvAdapter(path=handle).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Minimal Book by Jane Doe"
    assert unit.metadata["title"] == "Minimal Book"
    assert unit.metadata["author"] == "Jane Doe"
    assert "isbn" not in unit.metadata
    assert "my_rating" not in unit.metadata
    assert "review" not in unit.metadata
    assert unit.metadata["date_added"] == "2025-02-03T00:00:00+00:00"
    assert unit.tags == ["goodreads"]


def test_goodreads_reviews_csv_skips_blank_rows(tmp_path):
    path = tmp_path / "goodreads.csv"
    _write_csv(
        path,
        [
            {"Book Id": "", "Title": "", "Author": "", "Bookshelves": ""},
            {"Book Id": "2", "Title": "Kept", "Author": "Ada", "Bookshelves": ""},
        ],
    )

    result = GoodreadsReviewsCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].source_id == "goodreads_reviews_csv:2"


def test_goodreads_reviews_csv_parses_shelves_and_exclusive_shelf(tmp_path):
    path = tmp_path / "goodreads.csv"
    _write_csv(
        path,
        [
            {
                "Book Id": "1",
                "Title": "Tagged",
                "Author": "Ada",
                "Bookshelves": " Favorites, sci-fi;Favorites ",
                "Exclusive Shelf": "currently-reading",
            }
        ],
    )

    unit = GoodreadsReviewsCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.metadata["shelves"] == ["favorites", "sci-fi"]
    assert unit.tags == ["goodreads", "favorites", "sci-fi", "currently-reading"]
    assert "Shelves: favorites, sci-fi, currently-reading" in unit.content


def test_goodreads_reviews_csv_preserves_review_shelves_and_visibility_metadata(tmp_path):
    path = tmp_path / "goodreads.csv"
    _write_csv(
        path,
        [
            {
                "Book Id": "42",
                "Title": "Visibility",
                "Author": "Ada",
                "Review Shelves": "favorites | owned; favorites, research",
                "Exclusive Shelf": "read",
                "Spoiler": "yes",
                "Private Notes": "Private margin notes",
                "Owned Copies": "2",
                "Read Count": "3",
                "Date Added": "2026-05-01",
                "Date Updated": "2026-05-02T12:30:00Z",
            }
        ],
    )

    unit = GoodreadsReviewsCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.metadata["exclusive_shelf"] == "read"
    assert unit.metadata["shelves"] == ["favorites", "owned", "research"]
    assert unit.metadata["spoiler"] is True
    assert unit.metadata["private_notes"] == "Private margin notes"
    assert unit.metadata["owned_copies"] == 2
    assert unit.metadata["read_count"] == 3
    assert unit.metadata["date_added"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["date_updated"] == "2026-05-02T12:30:00+00:00"
    assert unit.tags == ["goodreads", "favorites", "owned", "research", "read"]


def test_goodreads_reviews_csv_normalizes_common_date_formats(tmp_path):
    path = tmp_path / "goodreads.csv"
    _write_csv(
        path,
        [
            {"Book Id": "1", "Title": "Slash", "Date Read": "2025/03/04"},
            {"Book Id": "2", "Title": "Iso", "Date Read": "2025-03-05T06:07:08Z"},
            {"Book Id": "3", "Title": "Us", "Date Added": "03/06/2025"},
            {"Book Id": "4", "Title": "Raw", "Date Read": "not a date"},
        ],
    )

    units = {unit.metadata["title"]: unit for unit in GoodreadsReviewsCsvAdapter(path=str(path)).ingest().units}

    assert units["Slash"].metadata["date_read"] == "2025-03-04T00:00:00+00:00"
    assert units["Iso"].metadata["date_read"] == "2025-03-05T06:07:08+00:00"
    assert units["Us"].metadata["date_added"] == "2025-03-06T00:00:00+00:00"
    assert units["Raw"].metadata["date_read"] == "not a date"


def test_goodreads_reviews_csv_source_ids_are_stable_without_book_id(tmp_path):
    path = tmp_path / "goodreads.csv"
    _write_csv(
        path,
        [
            {
                "Title": "Stable Book",
                "Author": "Ada Lovelace",
                "ISBN13": "9780000000002",
                "Year Published": "2025",
                "Date Added": "2025-01-01",
            }
        ],
    )

    first = GoodreadsReviewsCsvAdapter(path=str(path)).ingest().units[0]
    second = GoodreadsReviewsCsvAdapter(path=str(path)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("goodreads_reviews_csv:")
    assert first.source_id != "goodreads_reviews_csv:"
