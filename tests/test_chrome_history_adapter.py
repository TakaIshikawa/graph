from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone

from graph.adapters.chrome_history import ChromeHistoryAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def _chrome_time(value: datetime) -> int:
    epoch = datetime(1601, 1, 1, tzinfo=timezone.utc)
    return int((value - epoch).total_seconds() * 1_000_000)


def _create_history_db(path):
    with sqlite3.connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE urls (
                id INTEGER PRIMARY KEY,
                url TEXT,
                title TEXT,
                visit_count INTEGER,
                typed_count INTEGER,
                last_visit_time INTEGER,
                hidden INTEGER
            );
            CREATE TABLE visits (
                id INTEGER PRIMARY KEY,
                url INTEGER,
                visit_time INTEGER,
                from_visit INTEGER,
                transition INTEGER,
                visit_duration INTEGER
            );
            """
        )


def test_chrome_history_ingests_urls_and_visit_metadata(tmp_path):
    db = tmp_path / "History"
    _create_history_db(db)
    last_visit_time = _chrome_time(datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    transition = 1 | 0x02000000 | 0x10000000
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                1,
                "https://Example.com:443/research?q=graph#section",
                "Graph Research",
                7,
                2,
                last_visit_time,
                0,
            ),
        )
        conn.execute(
            "INSERT INTO visits VALUES (?, ?, ?, ?, ?, ?)",
            (10, 1, last_visit_time, 9, transition, 12_000_000),
        )

    result = ChromeHistoryAdapter(path=str(db)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    normalized_url = "https://example.com/research?q=graph"
    digest = hashlib.sha256(normalized_url.encode("utf-8")).hexdigest()[:24]
    assert unit.source_project == SourceProject.CHROME_HISTORY
    assert unit.source_entity_type == "page_visit"
    assert unit.source_id == f"chrome_history:{digest}"
    assert unit.title == "Graph Research"
    assert unit.content_type == ContentType.METADATA
    assert unit.tags == ["chrome", "browser_history"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.metadata["url"] == "https://Example.com:443/research?q=graph#section"
    assert unit.metadata["normalized_url"] == normalized_url
    assert unit.metadata["domain"] == "example.com"
    assert unit.metadata["visit_count"] == 7
    assert unit.metadata["typed_count"] == 2
    assert unit.metadata["last_visit_time"] == last_visit_time
    assert unit.metadata["last_visit_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["hidden"] is False
    assert unit.metadata["visit_duration"] == 12_000_000
    assert unit.metadata["from_visit"] == 9
    assert unit.metadata["transition"] == {
        "raw": transition,
        "core": 1,
        "type": "typed",
        "qualifiers": ["from_address_bar", "chain_start"],
    }
    assert "URL: https://example.com/research?q=graph" in unit.content
    assert "Transition: typed" in unit.content


def test_chrome_history_skips_internal_urls_by_default_and_can_include_them(tmp_path):
    db = tmp_path / "History"
    _create_history_db(db)
    visited_at = _chrome_time(datetime(2025, 1, 2, tzinfo=timezone.utc))
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)", (1, "", "", 1, 0, visited_at, 0))
        conn.execute(
            "INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, "chrome://settings", "Settings", 1, 0, visited_at, 0),
        )
        conn.execute(
            "INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)",
            (3, "about:blank", "Blank", 1, 0, visited_at, 0),
        )
        conn.execute(
            "INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)",
            (4, "https://example.com", "Example", 1, 0, visited_at, 0),
        )

    default_result = ChromeHistoryAdapter(path=str(db)).ingest()
    included_result = ChromeHistoryAdapter(path=str(db), include_internal_urls=True).ingest()

    assert [unit.metadata["normalized_url"] for unit in default_result.units] == ["https://example.com/"]
    assert sorted(unit.metadata["normalized_url"] for unit in included_result.units) == [
        "about:blank",
        "chrome:settings",
        "https://example.com/",
    ]


def test_chrome_history_filters_by_sync_state_and_entity_type(tmp_path):
    db = tmp_path / "History"
    _create_history_db(db)
    old_time = _chrome_time(datetime(2025, 1, 1, 10, tzinfo=timezone.utc))
    boundary_time = _chrome_time(datetime(2025, 1, 1, 11, tzinfo=timezone.utc))
    new_time = _chrome_time(datetime(2025, 1, 1, 12, tzinfo=timezone.utc))
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)", (1, "https://old.test", "Old", 1, 0, old_time, 0))
        conn.execute(
            "INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, "https://boundary.test", "Boundary", 1, 0, boundary_time, 0),
        )
        conn.execute("INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)", (3, "https://new.test", "New", 1, 0, new_time, 0))

    skipped = ChromeHistoryAdapter(path=str(db)).ingest(entity_types=["bookmark"])
    result = ChromeHistoryAdapter(path=str(db)).ingest(
        since=SyncState(
            source_project="chrome_history",
            source_entity_type="page_visit",
            last_sync_at=datetime(2025, 1, 1, 11, tzinfo=timezone.utc),
        )
    )

    assert skipped.units == []
    assert skipped.edges == []
    assert [unit.title for unit in result.units] == ["New"]


def test_chrome_history_source_id_is_stable(tmp_path):
    db = tmp_path / "History"
    _create_history_db(db)
    visited_at = _chrome_time(datetime(2025, 1, 1, tzinfo=timezone.utc))
    with sqlite3.connect(db) as conn:
        conn.execute(
            "INSERT INTO urls VALUES (?, ?, ?, ?, ?, ?, ?)",
            (1, "https://stable.test/path#fragment", "Stable", 1, 0, visited_at, 0),
        )

    first = ChromeHistoryAdapter(path=str(db)).ingest().units[0]
    second = ChromeHistoryAdapter(path=str(db)).ingest().units[0]

    assert first.source_id == second.source_id


def test_chrome_history_adapter_is_registered():
    assert "chrome_history" in list_adapters()
    adapter = get_adapter("chrome-history", path="/tmp/History")
    assert isinstance(adapter, ChromeHistoryAdapter)
    assert adapter.name == "chrome_history"
