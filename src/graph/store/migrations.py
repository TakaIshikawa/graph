"""SQLite schema creation."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 2

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_units (
    id TEXT PRIMARY KEY,
    source_project TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_entity_type TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'insight',
    metadata TEXT NOT NULL DEFAULT '{}',
    tags TEXT NOT NULL DEFAULT '[]',
    confidence REAL,
    utility_score REAL,
    embedding BLOB,
    embedding_updated_at TEXT,
    created_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(source_project, source_id, source_entity_type)
);

CREATE INDEX IF NOT EXISTS idx_ku_source
    ON knowledge_units(source_project, source_entity_type);
CREATE INDEX IF NOT EXISTS idx_ku_content_type
    ON knowledge_units(content_type);
CREATE INDEX IF NOT EXISTS idx_ku_created
    ON knowledge_units(created_at);

CREATE TABLE IF NOT EXISTS edges (
    id TEXT PRIMARY KEY,
    from_unit_id TEXT NOT NULL REFERENCES knowledge_units(id),
    to_unit_id TEXT NOT NULL REFERENCES knowledge_units(id),
    relation TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    source TEXT NOT NULL DEFAULT 'inferred',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    UNIQUE(from_unit_id, to_unit_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_unit_id);
CREATE INDEX IF NOT EXISTS idx_edges_to ON edges(to_unit_id);
CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges(relation);

CREATE TABLE IF NOT EXISTS sync_state (
    source_project TEXT NOT NULL,
    source_entity_type TEXT NOT NULL,
    last_sync_at TEXT NOT NULL,
    last_source_id TEXT,
    items_synced INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(source_project, source_entity_type)
);

CREATE TABLE IF NOT EXISTS saved_queries (
    name TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'fulltext',
    "limit" INTEGER NOT NULL DEFAULT 10,
    filters TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts
    USING fts5(unit_id UNINDEXED, title, content, tags);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    _ensure_column(conn, "knowledge_units", "embedding_updated_at", "TEXT")
    cursor = conn.execute("SELECT COUNT(*) FROM schema_version")
    if cursor.fetchone()[0] == 0:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
    else:
        conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
    conn.commit()


def _ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_type: str,
) -> None:
    columns = {
        row[1]
        for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
