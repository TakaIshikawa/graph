from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.goodreads_shelves_csv import GoodreadsShelvesCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_goodreads_shelves_csv_ingests_shelf_membership_metadata(tmp_path):
    export = tmp_path / "goodreads-shelves.csv"
    _write_csv(
        export,
        [
            {
                "Book Id": "123",
                "Title": "The Dispossessed",
                "Author": "Ursula K. Le Guin",
                "Exclusive Shelf": "Read",
                "Bookshelves": "Science Fiction, Favorites",
                "My Rating": "5",
                "Date Added": "2025/01/10",
                "Date Read": "2025/01/20",
                "URL": "https://www.goodreads.com/book/show/13651",
            }
        ],
    )

    result = GoodreadsShelvesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "goodreads_shelves_csv"
    assert unit.source_id.startswith("goodreads_shelves_csv:")
    assert unit.source_entity_type == "book_shelf_item"
    assert unit.title == "The Dispossessed by Ursula K. Le Guin [read, science fiction, favorites]"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Title: The Dispossessed" in unit.content
    assert "Author: Ursula K. Le Guin" in unit.content
    assert "Shelves: read, science fiction, favorites" in unit.content
    assert "My rating: 5/5" in unit.content
    assert "Date added: 2025-01-10" in unit.content
    assert "Date read: 2025-01-20" in unit.content
    assert "URL: https://www.goodreads.com/book/show/13651" in unit.content
    assert unit.metadata["book_id"] == "123"
    assert unit.metadata["title"] == "The Dispossessed"
    assert unit.metadata["author"] == "Ursula K. Le Guin"
    assert unit.metadata["exclusive_shelf"] == "read"
    assert unit.metadata["shelves"] == ["science fiction", "favorites"]
    assert unit.metadata["all_shelves"] == ["read", "science fiction", "favorites"]
    assert unit.metadata["rating"] == 5
    assert unit.metadata["my_rating"] == 5
    assert unit.metadata["date_added"] == "2025-01-10T00:00:00+00:00"
    assert unit.metadata["date_read"] == "2025-01-20T00:00:00+00:00"
    assert unit.metadata["url"] == "https://www.goodreads.com/book/show/13651"
    assert unit.metadata["source_file"] == "goodreads-shelves.csv"
    assert unit.metadata["row"]["Exclusive Shelf"] == "Read"
    assert unit.tags == ["goodreads", "book", "read", "science fiction", "favorites"]
    assert unit.created_at == datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 20, tzinfo=timezone.utc)


def test_goodreads_shelves_csv_directory_filters_and_distinct_duplicate_titles(tmp_path):
    _write_csv(
        tmp_path / "old.csv",
        [
            {
                "Title": "Same Book",
                "Author": "Ada",
                "Exclusive Shelf": "To Read",
                "Date Added": "2025-01-01",
            },
            {},
        ],
    )
    _write_csv(
        tmp_path / "new.csv",
        [
            {
                "Title": "Same Book",
                "Author": "Ada",
                "Exclusive Shelf": "Read",
                "Date Added": "2025-01-03",
            },
            {
                "Title": "Same Book",
                "Author": "Ada",
                "Exclusive Shelf": "Favorites",
                "Date Added": "2025-01-04",
            },
            {"Title": "   ", "Author": "", "Exclusive Shelf": ""},
        ],
    )
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = GoodreadsShelvesCsvAdapter(path=str(tmp_path))
    sync = SyncState(
        source_project="goodreads_shelves_csv",
        source_entity_type="book_shelf_item",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.metadata["exclusive_shelf"] for unit in first.units] == ["favorites", "read"]
    assert len({unit.source_id for unit in first.units}) == 2
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["book"]).units == []
