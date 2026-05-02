"""SQLite snapshot export helpers for graph data."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeEdge, KnowledgeUnit


def export_graph_sqlite(
    units: Iterable[KnowledgeUnit],
    edges: Iterable[KnowledgeEdge],
    path: str | Path,
    *,
    include_embeddings: bool = False,
) -> dict:
    """Write a portable SQLite snapshot of units, edges, tags, and metadata."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)

    all_units = list(units)
    all_edges = list(edges)
    exported_units = sorted(all_units, key=lambda unit: _text(unit.id))
    exported_edges = sorted(
        all_edges,
        key=lambda edge: (
            _text(edge.from_unit_id),
            _text(edge.to_unit_id),
            _enum_value(edge.relation),
            _text(edge.id),
        ),
    )

    with sqlite3.connect(output_path) as conn:
        _create_schema(conn, include_embeddings=include_embeddings)
        _insert_units(conn, exported_units, include_embeddings=include_embeddings)
        _insert_edges(conn, exported_edges)
        conn.commit()

    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "edges_scanned": len(all_edges),
        "edges_exported": len(exported_edges),
        "embeddings_included": include_embeddings,
        "bytes_written": output_path.stat().st_size,
    }


def _create_schema(conn: sqlite3.Connection, *, include_embeddings: bool) -> None:
    unit_embedding_column = ",\n        embedding_json TEXT" if include_embeddings else ""
    conn.executescript(
        f"""
        PRAGMA foreign_keys = ON;

        CREATE TABLE units (
            id TEXT PRIMARY KEY,
            source_project TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_entity_type TEXT NOT NULL,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            content_type TEXT NOT NULL,
            confidence REAL,
            utility_score REAL,
            created_at TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            updated_at TEXT NOT NULL{unit_embedding_column}
        );

        CREATE TABLE edges (
            id TEXT PRIMARY KEY,
            from_unit_id TEXT NOT NULL,
            to_unit_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            weight REAL NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE unit_tags (
            unit_id TEXT NOT NULL,
            tag TEXT NOT NULL,
            position INTEGER NOT NULL,
            PRIMARY KEY (unit_id, position),
            FOREIGN KEY (unit_id) REFERENCES units(id) ON DELETE CASCADE
        );

        CREATE TABLE metadata (
            owner_type TEXT NOT NULL,
            owner_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            PRIMARY KEY (owner_type, owner_id, key)
        );
        """
    )


def _insert_units(
    conn: sqlite3.Connection,
    units: Iterable[KnowledgeUnit],
    *,
    include_embeddings: bool,
) -> None:
    unit_columns = [
        "id",
        "source_project",
        "source_id",
        "source_entity_type",
        "title",
        "content",
        "content_type",
        "confidence",
        "utility_score",
        "created_at",
        "ingested_at",
        "updated_at",
    ]
    if include_embeddings:
        unit_columns.append("embedding_json")
    placeholders = ", ".join("?" for _ in unit_columns)
    column_names = ", ".join(unit_columns)

    for unit in units:
        values = [
            _text(unit.id),
            _enum_value(unit.source_project),
            _text(unit.source_id),
            _text(unit.source_entity_type),
            _text(unit.title),
            _text(unit.content),
            _enum_value(unit.content_type),
            unit.confidence,
            unit.utility_score,
            _datetime_text(unit.created_at),
            _datetime_text(unit.ingested_at),
            _datetime_text(unit.updated_at),
        ]
        if include_embeddings:
            values.append(_canonical_json(unit.embedding))
        conn.execute(f"INSERT INTO units ({column_names}) VALUES ({placeholders})", values)
        conn.executemany(
            "INSERT INTO unit_tags (unit_id, tag, position) VALUES (?, ?, ?)",
            [(_text(unit.id), _text(tag), position) for position, tag in enumerate(sorted(unit.tags))],
        )
        _insert_metadata(conn, "unit", _text(unit.id), unit.metadata)


def _insert_edges(conn: sqlite3.Connection, edges: Iterable[KnowledgeEdge]) -> None:
    for edge in edges:
        conn.execute(
            """
            INSERT INTO edges
                (id, from_unit_id, to_unit_id, relation, weight, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _text(edge.id),
                _text(edge.from_unit_id),
                _text(edge.to_unit_id),
                _enum_value(edge.relation),
                edge.weight,
                _enum_value(edge.source),
                _datetime_text(edge.created_at),
            ),
        )
        _insert_metadata(conn, "edge", _text(edge.id), edge.metadata)


def _insert_metadata(
    conn: sqlite3.Connection,
    owner_type: str,
    owner_id: str,
    metadata: Mapping[Any, Any],
) -> None:
    conn.executemany(
        "INSERT INTO metadata (owner_type, owner_id, key, value_json) VALUES (?, ?, ?, ?)",
        [
            (owner_type, owner_id, str(key), _canonical_json(value))
            for key, value in sorted(metadata.items(), key=_item_key)
        ],
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, BaseModel):
        return _json_value(value.model_dump())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=_item_key)}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    return str(value)


def _item_key(item: tuple[Any, Any]) -> str:
    return str(item[0])


def _enum_value(value: object) -> str:
    return _text(getattr(value, "value", value))


def _datetime_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return _text(value)


def _text(value: object) -> str:
    return str(value or "")
