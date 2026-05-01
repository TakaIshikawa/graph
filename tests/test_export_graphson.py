from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_graphson
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    *,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=content,
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
        confidence=0.8,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    *,
    source: EdgeSource = EdgeSource.INFERRED,
    weight: float = 1.0,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=source,
        weight=weight,
        metadata=metadata or {},
        created_at=EDGE_TIME,
    )


def load_graph(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_export_graphson_writes_nodes_edges_and_stats(tmp_path):
    path = tmp_path / "nested" / "graph.json"
    stats = export_graphson(
        [
            unit(
                "unit-a",
                "Alpha",
                "Alpha content",
                ["solar", "storage"],
                metadata={"reviewed_at": UNIT_TIME, "project": SourceProject.MAX},
            ),
            unit("unit-b", "Beta", "Beta content"),
        ],
        [
            edge(
                "edge-a",
                "unit-a",
                "unit-b",
                EdgeRelation.BUILDS_ON,
                source=EdgeSource.MANUAL,
                weight=0.75,
                metadata={"observed_at": EDGE_TIME, "notes": ["curated"]},
            )
        ],
        path,
    )

    graph = load_graph(path)

    assert stats == {
        "path": str(path),
        "node_count": 2,
        "edge_count": 1,
        "directed": True,
    }
    assert graph["directed"] is True
    assert graph["nodes"][0] == {
        "id": "unit-a",
        "label": "knowledge_unit",
        "properties": {
            "source_project": "max",
            "source_id": "source-unit-a",
            "source_entity_type": "insight",
            "title": "Alpha",
            "content": "Alpha content",
            "content_type": "insight",
            "metadata": {"project": "max", "reviewed_at": "2026-05-01T10:15:00+00:00"},
            "tags": ["solar", "storage"],
            "confidence": 0.8,
            "utility_score": 0.6,
            "created_at": "2026-05-01T10:15:00+00:00",
            "ingested_at": "2026-05-01T10:15:00+00:00",
            "updated_at": "2026-05-01T10:15:00+00:00",
        },
    }
    assert graph["edges"] == [
        {
            "id": "edge-a",
            "source": "unit-a",
            "target": "unit-b",
            "label": "builds_on",
            "properties": {
                "relation": "builds_on",
                "weight": 0.75,
                "source": "manual",
                "metadata": {"notes": ["curated"], "observed_at": "2026-05-01T12:30:00+00:00"},
                "created_at": "2026-05-01T12:30:00+00:00",
            },
        }
    ]


def test_export_graphson_sorts_output_deterministically(tmp_path):
    path = tmp_path / "graph.json"
    export_graphson(
        [
            unit("unit-b", "Beta", "Beta content"),
            unit("unit-a", "Alpha", "Alpha content"),
        ],
        [
            edge("edge-c", "unit-b", "unit-a", EdgeRelation.RELATES_TO),
            edge("edge-b", "unit-a", "unit-b", EdgeRelation.REFERENCES),
            edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        ],
        path,
    )

    graph = load_graph(path)

    assert [node["id"] for node in graph["nodes"]] == ["unit-a", "unit-b"]
    assert [item["id"] for item in graph["edges"]] == ["edge-a", "edge-b", "edge-c"]
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_export_graphson_supports_undirected_empty_export(tmp_path):
    path = tmp_path / "empty" / "graph.json"
    stats = export_graphson([], [], path, directed=False)

    assert stats == {
        "path": str(path),
        "node_count": 0,
        "edge_count": 0,
        "directed": False,
    }
    assert load_graph(path) == {"directed": False, "edges": [], "nodes": []}
