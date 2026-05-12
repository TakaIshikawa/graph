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
    assert unit.metadata["first_visit_at"] == "2025-01-01T00:01:40+00:00"
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


def test_firefox_places_aggregates_history_visit_dates_and_transitions(tmp_path):
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
                visit_date INTEGER,
                transition INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO moz_places VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "https://transitions.test", "Transitions", 5, 80, 0, 1_735_689_600_000_000),
        )
        conn.executemany(
            "INSERT INTO moz_historyvisits VALUES (?, ?, ?, ?)",
            [
                (1, 1, 1_735_689_600_000_000, 1),
                (2, 1, 1_735_693_200_000_000, 2),
                (3, 1, 1_735_696_800_000_000, 1),
            ],
        )

    result = FirefoxPlacesAdapter(path=str(db)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.created_at == datetime(2025, 1, 1, 2, 0, tzinfo=timezone.utc)
    assert unit.metadata["history_visit_count"] == 3
    assert unit.metadata["first_visit_at"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["last_visit_at"] == "2025-01-01T02:00:00+00:00"
    assert unit.metadata["transition_counts"] == {"1": 2, "2": 1}


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


def test_firefox_places_emits_search_terms_from_search_urls(tmp_path):
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
        conn.executemany(
            "INSERT INTO moz_places VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (1, "https://www.google.com/search?q=Solar+Storage", "Google", 2, 10, 0, 1_735_689_600_000_000),
                (2, "https://duckduckgo.com/?q=solar%20storage", "DuckDuckGo", 4, 10, 0, 1_735_693_200_000_000),
            ],
        )
        conn.executemany(
            "INSERT INTO moz_historyvisits VALUES (?, ?, ?)",
            [
                (1, 1, 1_735_689_600_000_000),
                (2, 2, 1_735_693_200_000_000),
                (3, 2, 1_735_696_800_000_000),
            ],
        )

    result = FirefoxPlacesAdapter(path=str(db)).ingest(entity_types=["search_term"])

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_entity_type == "search_term"
    assert unit.title == "solar storage"
    assert unit.source_id.startswith("firefox_places:search:")
    assert unit.metadata == {
        "query": "solar storage",
        "visit_count": 3,
        "first_seen_at": "2025-01-01T00:00:00+00:00",
        "last_seen_at": "2025-01-01T02:00:00+00:00",
        "source_table": "moz_places",
        "source_file": "places.sqlite",
    }
    assert unit.tags == ["firefox", "search"]


def test_firefox_places_emits_search_terms_from_moz_keywords(tmp_path):
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
            CREATE TABLE moz_keywords (
                id INTEGER PRIMARY KEY,
                keyword TEXT,
                place_id INTEGER
            );
            """
        )
        conn.execute(
            "INSERT INTO moz_places VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "https://example.com/search?q=%s", "Search", 1, 10, 0, 1_735_689_600_000_000),
        )
        conn.execute(
            "INSERT INTO moz_historyvisits VALUES (?, ?, ?)",
            (1, 1, 1_735_689_600_000_000),
        )
        conn.execute("INSERT INTO moz_keywords VALUES (?, ?, ?)", (1, "Paper Search", 1))

    unit = FirefoxPlacesAdapter(path=str(db)).ingest(entity_types=["search_term"]).units[0]

    assert unit.title == "paper search"
    assert unit.metadata["source_table"] == "moz_keywords"
    assert unit.metadata["visit_count"] == 1


def test_firefox_places_handles_missing_search_tables_gracefully(tmp_path):
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
            (1, "https://example.com/page", "Page", 1, 10, 0, 1_735_689_600_000_000),
        )

    result = FirefoxPlacesAdapter(path=str(db)).ingest(entity_types=["search_term"])

    assert result.units == []


def test_firefox_places_adapter_is_registered():
    assert "firefox_places" in list_adapters()
    assert get_adapter("firefox_places", path="/tmp/places.sqlite").name == "firefox_places"
