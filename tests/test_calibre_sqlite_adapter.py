from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from graph.adapters.calibre_sqlite import CalibreSqliteAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_calibre_sqlite_ingests_book_metadata(tmp_path):
    db_path = _calibre_db(tmp_path)

    result = CalibreSqliteAdapter(path=str(db_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "calibre_sqlite"
    assert unit.source_id == "calibre_sqlite:42"
    assert unit.source_entity_type == "book"
    assert unit.title == "The Example Book by Ada Lovelace, Grace Hopper"
    assert unit.tags == ["computing", "history"]
    assert unit.metadata["book_id"] == 42
    assert unit.metadata["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert unit.metadata["tags"] == ["computing", "history"]
    assert unit.metadata["formats"] == ["EPUB", "PDF"]
    assert unit.metadata["identifiers"] == {
        "doi": "10.1234/example",
        "isbn": "9780000000001",
    }
    assert unit.metadata["publisher"] == "Example Press"
    assert unit.metadata["rating"] == 8
    assert unit.metadata["publication_date"] == "2020-01-02T00:00:00+00:00"
    assert unit.metadata["added_at"] == "2024-03-04T05:06:07+00:00"
    assert unit.metadata["updated_at"] == "2024-04-05T06:07:08+00:00"
    assert unit.metadata["library_path"] == str(tmp_path / "Ada Lovelace" / "The Example Book (42)")
    assert unit.metadata["comments"] == "Important notes\nSecond paragraph"
    assert "Identifiers: doi:10.1234/example, isbn:9780000000001" in unit.content


def test_calibre_sqlite_accepts_library_directory_and_since_filters_on_updated_at(tmp_path):
    db_path = _calibre_db(tmp_path)
    _insert_book(
        db_path,
        book_id=43,
        title="New Book",
        timestamp="2024-02-01T00:00:00+00:00",
        last_modified="2024-06-01T00:00:00+00:00",
        path="New Book (43)",
    )

    result = CalibreSqliteAdapter(path=str(tmp_path)).ingest(
        since=SyncState(
            source_project="calibre_sqlite",
            source_entity_type="book",
            last_sync_at=datetime(2024, 5, 1, tzinfo=timezone.utc),
        )
    )

    assert [unit.source_id for unit in result.units] == ["calibre_sqlite:43"]


def test_calibre_sqlite_since_uses_timestamp_when_last_modified_missing(tmp_path):
    db_path = _minimal_calibre_db(tmp_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO books (id, title, timestamp) VALUES (?, ?, ?)",
            (1, "Old Book", "2024-01-01T00:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO books (id, title, timestamp) VALUES (?, ?, ?)",
            (2, "New Book", "2024-03-01T00:00:00+00:00"),
        )

    result = CalibreSqliteAdapter(path=str(db_path)).ingest(
        since=SyncState(
            source_project="calibre_sqlite",
            source_entity_type="book",
            last_sync_at=datetime(2024, 2, 1, tzinfo=timezone.utc),
        )
    )

    assert [unit.source_id for unit in result.units] == ["calibre_sqlite:2"]


def test_calibre_sqlite_registry_lookup(tmp_path):
    adapter = get_adapter("calibre_sqlite", path=str(tmp_path / "metadata.db"))

    assert isinstance(adapter, CalibreSqliteAdapter)
    assert adapter.name == "calibre_sqlite"


def _calibre_db(tmp_path):
    db_path = tmp_path / "metadata.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE books (
                id INTEGER PRIMARY KEY,
                title TEXT,
                sort TEXT,
                timestamp TEXT,
                pubdate TEXT,
                path TEXT,
                uuid TEXT,
                has_cover BOOL,
                last_modified TEXT,
                author_sort TEXT,
                isbn TEXT
            );
            CREATE TABLE authors (id INTEGER PRIMARY KEY, name TEXT, sort TEXT, link TEXT);
            CREATE TABLE books_authors_link (id INTEGER PRIMARY KEY, book INTEGER, author INTEGER);
            CREATE TABLE tags (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_tags_link (id INTEGER PRIMARY KEY, book INTEGER, tag INTEGER);
            CREATE TABLE identifiers (id INTEGER PRIMARY KEY, book INTEGER, type TEXT, val TEXT);
            CREATE TABLE comments (id INTEGER PRIMARY KEY, book INTEGER, text TEXT);
            CREATE TABLE publishers (id INTEGER PRIMARY KEY, name TEXT);
            CREATE TABLE books_publishers_link (id INTEGER PRIMARY KEY, book INTEGER, publisher INTEGER);
            CREATE TABLE ratings (id INTEGER PRIMARY KEY, rating INTEGER);
            CREATE TABLE books_ratings_link (id INTEGER PRIMARY KEY, book INTEGER, rating INTEGER);
            CREATE TABLE data (id INTEGER PRIMARY KEY, book INTEGER, format TEXT, name TEXT, uncompressed_size INTEGER);
            """
        )
        _insert_book(
            db_path,
            book_id=42,
            title="The Example Book",
            timestamp="2024-03-04T05:06:07+00:00",
            last_modified="2024-04-05T06:07:08+00:00",
            path="Ada Lovelace/The Example Book (42)",
            pubdate="2020-01-02T00:00:00+00:00",
            uuid="book-uuid",
            author_sort="Lovelace, Ada",
            isbn="9780000000001",
        )
        conn.executemany("INSERT INTO authors (id, name) VALUES (?, ?)", [(1, "Ada Lovelace"), (2, "Grace Hopper")])
        conn.executemany("INSERT INTO books_authors_link (id, book, author) VALUES (?, ?, ?)", [(1, 42, 1), (2, 42, 2)])
        conn.executemany("INSERT INTO tags (id, name) VALUES (?, ?)", [(1, "computing"), (2, "history")])
        conn.executemany("INSERT INTO books_tags_link (id, book, tag) VALUES (?, ?, ?)", [(1, 42, 1), (2, 42, 2)])
        conn.executemany(
            "INSERT INTO identifiers (book, type, val) VALUES (?, ?, ?)",
            [(42, "doi", "10.1234/example")],
        )
        conn.execute(
            "INSERT INTO comments (book, text) VALUES (?, ?)",
            (42, "<p>Important notes</p><p>Second paragraph</p>"),
        )
        conn.execute("INSERT INTO publishers (id, name) VALUES (?, ?)", (1, "Example Press"))
        conn.execute("INSERT INTO books_publishers_link (book, publisher) VALUES (?, ?)", (42, 1))
        conn.execute("INSERT INTO ratings (id, rating) VALUES (?, ?)", (1, 8))
        conn.execute("INSERT INTO books_ratings_link (book, rating) VALUES (?, ?)", (42, 1))
        conn.executemany("INSERT INTO data (book, format) VALUES (?, ?)", [(42, "EPUB"), (42, "PDF")])
    return db_path


def _minimal_calibre_db(tmp_path):
    db_path = tmp_path / "metadata.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE books (id INTEGER PRIMARY KEY, title TEXT, timestamp TEXT)")
    return db_path


def _insert_book(
    db_path,
    *,
    book_id,
    title,
    timestamp,
    last_modified,
    path,
    pubdate="0101-01-01T00:00:00+00:00",
    uuid="",
    author_sort="",
    isbn="",
):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO books (
                id, title, sort, timestamp, pubdate, path, uuid, has_cover, last_modified, author_sort, isbn
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (book_id, title, title, timestamp, pubdate, path, uuid, 0, last_modified, author_sort, isbn),
        )
