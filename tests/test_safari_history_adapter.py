from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter, list_adapters
from graph.adapters.safari_history import SafariHistoryAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def _safari_time(value: datetime) -> float:
    epoch = datetime(2001, 1, 1, tzinfo=timezone.utc)
    return (value - epoch).total_seconds()


def _create_history_db(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE history_items (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT,
                domain_expansion TEXT,
                visit_count INTEGER
            );
            CREATE TABLE history_visits (
                id INTEGER PRIMARY KEY,
                history_item INTEGER,
                visit_time REAL,
                title TEXT,
                load_successful INTEGER,
                http_non_get INTEGER,
                synthesized INTEGER,
                redirect_source INTEGER,
                redirect_destination INTEGER,
                origin INTEGER,
                attributes INTEGER,
                score REAL
            );
            """
        )


def test_safari_history_ingests_one_unit_per_visit_with_metadata(tmp_path):
    db = tmp_path / "History.db"
    _create_history_db(db)
    first_visit = _safari_time(datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    second_visit = _safari_time(datetime(2025, 1, 2, 4, 5, 6, tzinfo=timezone.utc))
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO history_items VALUES (?, ?, ?, ?, ?)",
            (1, "https://example.com/research", "Item Title", "example.com", 2),
        )
        conn.execute(
            "INSERT INTO history_visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (10, 1, first_visit, "Visit Title", 1, 0, 0, None, None, 3, 17, 42.5),
        )
        conn.execute(
            "INSERT INTO history_visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (11, 1, second_visit, "Second Visit", 1, 0, 0, 10, None, 3, 18, 43.5),
        )

    result = SafariHistoryAdapter(path=str(db)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_project == SourceProject.SAFARI_HISTORY
    assert unit.source_entity_type == "page_visit"
    assert unit.source_id.startswith("safari_history:")
    assert unit.title == "Visit Title"
    assert unit.content_type == ContentType.METADATA
    assert unit.tags == ["safari", "browser_history"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.metadata["url"] == "https://example.com/research"
    assert unit.metadata["title"] == "Visit Title"
    assert unit.metadata["visited_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["browser"] == "safari"
    assert unit.metadata["source_name"] == "Safari History"
    assert unit.metadata["source_file"] == "History.db"
    assert unit.metadata["history_item_id"] == 1
    assert unit.metadata["visit_id"] == 10
    assert unit.metadata["domain_expansion"] == "example.com"
    assert unit.metadata["visit_count"] == 2
    assert unit.metadata["load_successful"] is True
    assert unit.metadata["http_non_get"] is False
    assert unit.metadata["synthesized"] is False
    assert unit.metadata["origin"] == 3
    assert unit.metadata["attributes"] == 17
    assert unit.metadata["score"] == 42.5
    assert "URL: https://example.com/research" in unit.content
    assert "Visited: 2025-01-02T03:04:05+00:00" in unit.content


def test_safari_history_falls_back_to_item_title_then_url(tmp_path):
    db = tmp_path / "History.db"
    _create_history_db(db)
    first_visit = _safari_time(datetime(2025, 1, 2, tzinfo=timezone.utc))
    second_visit = _safari_time(datetime(2025, 1, 2, 0, 1, tzinfo=timezone.utc))
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO history_items VALUES (?, ?, ?, ?, ?)",
            (1, "https://item-title.test", "Item Title", "item-title.test", 1),
        )
        conn.execute(
            "INSERT INTO history_items VALUES (?, ?, ?, ?, ?)",
            (2, "https://url-fallback.test", "", "url-fallback.test", 1),
        )
        conn.execute(
            "INSERT INTO history_visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (10, 1, first_visit, "", 1, 0, 0, None, None, None, None, None),
        )
        conn.execute(
            "INSERT INTO history_visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (11, 2, second_visit, None, 1, 0, 0, None, None, None, None, None),
        )

    result = SafariHistoryAdapter(path=str(db)).ingest()

    assert [unit.title for unit in result.units] == ["Item Title", "https://url-fallback.test"]


def test_safari_history_empty_database_and_filters(tmp_path):
    db = tmp_path / "History.db"
    _create_history_db(db)
    skipped = SafariHistoryAdapter(path=str(db)).ingest(entity_types=["bookmark"])
    result = SafariHistoryAdapter(path=str(db)).ingest()

    assert skipped.units == []
    assert skipped.edges == []
    assert result.units == []
    assert result.edges == []


def test_safari_history_filters_by_sync_state(tmp_path):
    db = tmp_path / "History.db"
    _create_history_db(db)
    old_time = _safari_time(datetime(2025, 1, 1, 10, tzinfo=timezone.utc))
    boundary_time = _safari_time(datetime(2025, 1, 1, 11, tzinfo=timezone.utc))
    new_time = _safari_time(datetime(2025, 1, 1, 12, tzinfo=timezone.utc))
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO history_items VALUES (?, ?, ?, ?, ?)", (1, "https://old.test", "Old", "old.test", 1))
        conn.execute(
            "INSERT INTO history_items VALUES (?, ?, ?, ?, ?)",
            (2, "https://boundary.test", "Boundary", "boundary.test", 1),
        )
        conn.execute("INSERT INTO history_items VALUES (?, ?, ?, ?, ?)", (3, "https://new.test", "New", "new.test", 1))
        conn.execute(
            "INSERT INTO history_visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (10, 1, old_time, "Old", 1, 0, 0, None, None, None, None, None),
        )
        conn.execute(
            "INSERT INTO history_visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (11, 2, boundary_time, "Boundary", 1, 0, 0, None, None, None, None, None),
        )
        conn.execute(
            "INSERT INTO history_visits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (12, 3, new_time, "New", 1, 0, 0, None, None, None, None, None),
        )

    result = SafariHistoryAdapter(path=str(db)).ingest(
        since=SyncState(
            source_project="safari_history",
            source_entity_type="page_visit",
            last_sync_at=datetime(2025, 1, 1, 11, tzinfo=timezone.utc),
        )
    )

    assert [unit.title for unit in result.units] == ["New"]


def test_safari_history_adapter_is_registered():
    assert "safari_history" in list_adapters()
    adapter = get_adapter("safari-history", path="/tmp/History.db")
    assert isinstance(adapter, SafariHistoryAdapter)
    assert adapter.name == "safari_history"
