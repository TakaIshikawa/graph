from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from graph.adapters.sqlite_query_log import SqliteQueryLogAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_sqlite_query_log_ingests_queries_from_database(tmp_path):
    db_path = tmp_path / "queries.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE searches (search_text TEXT, searched_at INTEGER, hits INTEGER)")
    conn.execute(
        "INSERT INTO searches (search_text, searched_at, hits) VALUES (?, ?, ?)",
        ("sqlite query logs", 1_710_000_000, 7),
    )
    conn.commit()
    conn.close()

    result = SqliteQueryLogAdapter(
        db_path=str(db_path),
        table="searches",
        query_column="search_text",
        created_column="searched_at",
        result_count_column="hits",
    ).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.SQLITE_QUERY_LOG
    assert unit.source_entity_type == "query_log"
    assert unit.source_id.startswith("sqlite_query_log:searches:1:")
    assert unit.title == "sqlite query logs"
    assert unit.content == "sqlite query logs"
    assert unit.content_type == ContentType.METADATA
    assert unit.created_at == datetime.fromtimestamp(1_710_000_000, tz=timezone.utc)
    assert unit.updated_at == unit.created_at
    assert unit.metadata == {
        "table": "searches",
        "query_column": "search_text",
        "created_column": "searched_at",
        "rowid": 1,
        "raw_created_at": 1_710_000_000,
        "result_count_column": "hits",
        "result_count": 7,
    }


def test_sqlite_query_log_parses_unix_timestamps_for_since_filtering(tmp_path):
    db_path = tmp_path / "queries.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE queries (query TEXT, created_at INTEGER)")
    conn.execute(
        "INSERT INTO queries (query, created_at) VALUES (?, ?)",
        ("old query", 1_710_000_000),
    )
    conn.execute(
        "INSERT INTO queries (query, created_at) VALUES (?, ?)",
        ("new query", 1_710_086_400_000),
    )
    conn.commit()
    conn.close()

    result = SqliteQueryLogAdapter(db_path=str(db_path)).ingest(
        since=SyncState(
            source_project="sqlite_query_log",
            source_entity_type="query_log",
            last_sync_at=datetime(2024, 3, 10, tzinfo=timezone.utc),
        )
    )

    assert [unit.title for unit in result.units] == ["new query"]
    assert result.units[0].created_at == datetime(2024, 3, 10, 16, tzinfo=timezone.utc)


def test_sqlite_query_log_missing_optional_result_count_does_not_fail(tmp_path):
    db_path = tmp_path / "queries.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE queries (query TEXT, created_at TEXT)")
    conn.execute(
        "INSERT INTO queries (query, created_at) VALUES (?, ?)",
        ("adapter registry", "2025-01-02T03:04:05Z"),
    )
    conn.commit()
    conn.close()

    result = SqliteQueryLogAdapter(db_path=str(db_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "adapter registry"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert "result_count" not in unit.metadata
    assert "result_count_column" not in unit.metadata


def test_sqlite_query_log_respects_entity_type_filter(tmp_path):
    db_path = tmp_path / "queries.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE queries (query TEXT, created_at TEXT)")
    conn.execute(
        "INSERT INTO queries (query, created_at) VALUES (?, ?)",
        ("ignored", "2025-01-02T03:04:05Z"),
    )
    conn.commit()
    conn.close()

    result = SqliteQueryLogAdapter(db_path=str(db_path)).ingest(entity_types=["other_entity"])

    assert result.units == []
    assert result.edges == []
