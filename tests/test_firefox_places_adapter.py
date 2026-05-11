from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from graph.adapters.firefox_places import FirefoxPlacesAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject


def test_firefox_places_ingests_history_and_metadata(tmp_path):
    db = tmp_path / "places.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE moz_places (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT,
                visit_count INTEGER,
                frecency INTEGER,
                typed INTEGER,
                last_visit_date INTEGER
            );
            CREATE TABLE moz_historyvisits (
                id INTEGER PRIMARY KEY,
                place_id INTEGER,
                visit_date INTEGER
            );
            CREATE TABLE moz_bookmarks (
                id INTEGER PRIMARY KEY,
                fk INTEGER,
                type INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO moz_places VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "https://example.com", "Example", 3, 120, 1, 1_735_689_600_000_000),
        )
        conn.execute(
            "INSERT INTO moz_historyvisits VALUES (?, ?, ?)",
            (1, 1, 1_735_689_700_000_000),
        )
        conn.execute("INSERT INTO moz_bookmarks VALUES (?, ?, ?)", (1, 1, 1))

    result = FirefoxPlacesAdapter(path=str(db)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.FIREFOX_PLACES
    assert unit.source_entity_type == "page_visit"
    assert unit.source_id.startswith("firefox_places:")
    assert unit.title == "Example"
    assert unit.content == "https://example.com"
    assert unit.created_at == datetime(2025, 1, 1, 0, 1, 40, tzinfo=timezone.utc)
    assert unit.metadata["url"] == "https://example.com"
    assert unit.metadata["visit_count"] == 3
    assert unit.metadata["history_visit_count"] == 1
    assert unit.metadata["frecency"] == 120
    assert unit.metadata["typed"] is True
    assert unit.metadata["bookmarked"] is True


def test_firefox_places_opens_database_read_only(tmp_path):
    db = tmp_path / "places.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE moz_places (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT,
                visit_count INTEGER,
                frecency INTEGER,
                typed INTEGER,
                last_visit_date INTEGER
            );
            CREATE TABLE moz_historyvisits (
                id INTEGER PRIMARY KEY,
                place_id INTEGER,
                visit_date INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO moz_places VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "https://readonly.test", "Readonly", 1, 10, 0, 1_735_689_600_000_000),
        )

    before = db.stat().st_mtime_ns
    result = FirefoxPlacesAdapter(path=str(db)).ingest()
    after = db.stat().st_mtime_ns

    assert len(result.units) == 1
    assert before == after


def test_firefox_places_ingests_bookmark_without_visit(tmp_path):
    db = tmp_path / "places.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE moz_places (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT,
                visit_count INTEGER,
                frecency INTEGER,
                typed INTEGER,
                last_visit_date INTEGER
            );
            CREATE TABLE moz_bookmarks (
                id INTEGER PRIMARY KEY,
                fk INTEGER,
                type INTEGER,
                dateAdded INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO moz_places VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "https://bookmark.test", "Saved only", 0, 20, 0, None),
        )
        conn.execute(
            "INSERT INTO moz_bookmarks VALUES (?, ?, ?, ?)",
            (1, 1, 1, 1_735_689_600_000_000),
        )

    result = FirefoxPlacesAdapter(path=str(db)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Saved only"
    assert unit.metadata["bookmarked"] is True
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_firefox_places_uses_url_for_missing_title(tmp_path):
    db = tmp_path / "places.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE moz_places (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT,
                visit_count INTEGER,
                frecency INTEGER,
                typed INTEGER,
                last_visit_date INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO moz_places VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "https://untitled.test/path", "", 1, 10, 0, 1_735_689_600_000_000),
        )

    result = FirefoxPlacesAdapter(path=str(db)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "https://untitled.test/path"


def test_firefox_places_adapter_is_registered():
    assert "firefox_places" in list_adapters()
    assert get_adapter("firefox_places", path="/tmp/places.sqlite").name == "firefox_places"
