from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from graph.export import export_graph_sqlite
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


SNAPSHOT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    embedding: list[float] | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.FINDING,
        metadata=metadata or {},
        tags=tags or [],
        confidence=0.8,
        utility_score=0.6,
        embedding=embedding,
        created_at=SNAPSHOT_TIME,
        ingested_at=SNAPSHOT_TIME,
        updated_at=SNAPSHOT_TIME,
    )


def edge(edge_id: str, from_unit_id: str, to_unit_id: str, *, metadata: dict | None = None) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=EdgeRelation.REFERENCES,
        weight=0.75,
        source=EdgeSource.MANUAL,
        metadata=metadata or {},
        created_at=SNAPSHOT_TIME,
    )


def rows(path, query: str):
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query)]


def test_export_graph_sqlite_writes_readable_snapshot_tables_and_stats(tmp_path):
    path = tmp_path / "nested" / "graph.sqlite"

    stats = export_graph_sqlite(
        [
            unit(
                "unit-b",
                tags=["zeta", "alpha"],
                metadata={
                    "nested": {"b": 2, "a": SNAPSHOT_TIME},
                    "source": SourceProject.RAINDROP,
                },
            ),
            unit("unit-a"),
        ],
        [edge("edge-1", "unit-a", "unit-b", metadata={"score": 0.5})],
        path,
    )

    assert stats == {
        "path": str(path),
        "units_scanned": 2,
        "units_exported": 2,
        "edges_scanned": 1,
        "edges_exported": 1,
        "embeddings_included": False,
        "bytes_written": path.stat().st_size,
    }
    assert rows(path, "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name") == [
        {"name": "edges"},
        {"name": "metadata"},
        {"name": "unit_tags"},
        {"name": "units"},
    ]
    assert rows(path, "SELECT id, source_project, content_type, created_at FROM units ORDER BY id") == [
        {
            "id": "unit-a",
            "source_project": "max",
            "content_type": "finding",
            "created_at": "2026-05-01T10:15:00+00:00",
        },
        {
            "id": "unit-b",
            "source_project": "max",
            "content_type": "finding",
            "created_at": "2026-05-01T10:15:00+00:00",
        },
    ]
    assert rows(path, "SELECT id, relation, source, weight FROM edges") == [
        {"id": "edge-1", "relation": "references", "source": "manual", "weight": 0.75}
    ]
    assert rows(path, "SELECT unit_id, tag, position FROM unit_tags ORDER BY unit_id, position") == [
        {"unit_id": "unit-b", "tag": "alpha", "position": 0},
        {"unit_id": "unit-b", "tag": "zeta", "position": 1},
    ]
    metadata_rows = rows(
        path,
        "SELECT owner_type, owner_id, key, value_json FROM metadata ORDER BY owner_type, owner_id, key",
    )
    assert metadata_rows == [
        {"owner_type": "edge", "owner_id": "edge-1", "key": "score", "value_json": "0.5"},
        {
            "owner_type": "unit",
            "owner_id": "unit-b",
            "key": "nested",
            "value_json": '{"a":"2026-05-01T10:15:00+00:00","b":2}',
        },
        {"owner_type": "unit", "owner_id": "unit-b", "key": "source", "value_json": '"raindrop"'},
    ]
    assert json.loads(metadata_rows[1]["value_json"]) == {
        "a": "2026-05-01T10:15:00+00:00",
        "b": 2,
    }


def test_export_graph_sqlite_replaces_existing_file_deterministically(tmp_path):
    path = tmp_path / "graph.sqlite"

    export_graph_sqlite([unit("unit-old")], [edge("edge-old", "unit-old", "unit-old")], path)
    export_graph_sqlite([unit("unit-new")], [], path)

    assert rows(path, "SELECT id FROM units") == [{"id": "unit-new"}]
    assert rows(path, "SELECT id FROM edges") == []


def test_export_graph_sqlite_excludes_embeddings_by_default(tmp_path):
    path = tmp_path / "graph.sqlite"

    export_graph_sqlite([unit("unit-a", embedding=[0.1, 0.2])], [], path)

    assert "embedding_json" not in [
        row["name"] for row in rows(path, "PRAGMA table_info(units)")
    ]


def test_export_graph_sqlite_includes_embeddings_when_requested(tmp_path):
    path = tmp_path / "graph.sqlite"

    export_graph_sqlite([unit("unit-a", embedding=[0.1, 0.2])], [], path, include_embeddings=True)

    assert rows(path, "SELECT id, embedding_json FROM units") == [
        {"id": "unit-a", "embedding_json": "[0.1,0.2]"}
    ]
