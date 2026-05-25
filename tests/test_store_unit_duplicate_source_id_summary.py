from __future__ import annotations

import sqlite3

from graph.store.unit_duplicate_source_id_summary import summarize_unit_duplicate_source_ids


def connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE knowledge_units (
            id TEXT PRIMARY KEY,
            source_project TEXT NOT NULL,
            source_entity_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            title TEXT NOT NULL,
            updated_at TEXT
        )
        """
    )
    return conn


def insert_unit(
    conn: sqlite3.Connection,
    unit_id: str,
    source_project: str,
    source_entity_type: str,
    source_id: str,
    title: str,
    updated_at: str | None,
) -> None:
    conn.execute(
        """
        INSERT INTO knowledge_units
            (id, source_project, source_entity_type, source_id, title, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (unit_id, source_project, source_entity_type, source_id, title, updated_at),
    )
    conn.commit()


def test_summarize_unit_duplicate_source_ids_groups_by_full_source_identity():
    conn = connection()
    insert_unit(conn, "b", "max", "page", "shared", "Second", "2026-01-02T00:00:00+00:00")
    insert_unit(conn, "a", "max", "page", "shared", "First", "2026-01-01T00:00:00+00:00")
    insert_unit(conn, "c", "max", "comment", "shared", "Different entity", "2026-01-03T00:00:00+00:00")
    insert_unit(conn, "d", "csv", "page", "shared", "Different project", "2026-01-04T00:00:00+00:00")

    summary = summarize_unit_duplicate_source_ids(conn)

    assert summary["duplicate_count"] == 1
    assert summary["rows"] == [
        {
            "source_project": "max",
            "source_entity_type": "page",
            "source_id": "shared",
            "unit_count": 2,
            "unit_ids": ["a", "b"],
            "titles": ["First", "Second"],
            "latest_updated_at": "2026-01-02T00:00:00+00:00",
        }
    ]
    assert summary["duplicate_groups"] == summary["rows"]


def test_summarize_unit_duplicate_source_ids_handles_missing_updated_at():
    conn = connection()
    insert_unit(conn, "a", "max", "page", "shared", "First", None)
    insert_unit(conn, "b", "max", "page", "shared", "Second", "not-a-date")

    summary = summarize_unit_duplicate_source_ids(conn)

    assert summary["rows"][0]["latest_updated_at"] is None
