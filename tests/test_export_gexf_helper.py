from __future__ import annotations

import json
from datetime import datetime, timezone

import networkx as nx

from graph.export import export_graph_gexf
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
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
        content=f"{title} content",
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


def test_export_graph_gexf_writes_parseable_graph_and_summary(tmp_path):
    path = tmp_path / "nested" / "graph.gexf"
    stats = export_graph_gexf(
        [
            unit(
                "unit-a",
                "Alpha",
                ["solar", "storage"],
                metadata={"reviewed_at": UNIT_TIME, "rank": 1},
            ),
            unit("unit-b", "Beta", ["grid"]),
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

    assert stats == {
        "path": str(path),
        "units_exported": 2,
        "edges_exported": 1,
        "skipped_edges": [],
    }

    exported = nx.read_gexf(path)
    assert set(exported.nodes) == {"unit-a", "unit-b"}
    assert exported.nodes["unit-a"]["title"] == "Alpha"
    assert exported.nodes["unit-a"]["source_project"] == "max"
    assert exported.nodes["unit-a"]["source_id"] == "source-unit-a"
    assert exported.nodes["unit-a"]["source_entity_type"] == "insight"
    assert exported.nodes["unit-a"]["content_type"] == "insight"
    assert exported.nodes["unit-a"]["tags"] == "solar,storage"
    assert json.loads(exported.nodes["unit-a"]["metadata"]) == {
        "rank": 1,
        "reviewed_at": "2026-05-01T10:15:00+00:00",
    }
    assert float(exported.nodes["unit-a"]["confidence"]) == 0.8
    assert float(exported.nodes["unit-a"]["utility_score"]) == 0.6
    assert exported.nodes["unit-a"]["created_at"] == UNIT_TIME.isoformat()
    assert exported.nodes["unit-a"]["updated_at"] == UNIT_TIME.isoformat()

    edge_data = exported.get_edge_data("unit-a", "unit-b")
    assert edge_data["relation"] == "builds_on"
    assert edge_data["source"] == "manual"
    assert float(edge_data["weight"]) == 0.75
    assert json.loads(edge_data["metadata"]) == {
        "notes": ["curated"],
        "observed_at": "2026-05-01T12:30:00+00:00",
    }
    assert edge_data["created_at"] == EDGE_TIME.isoformat()


def test_export_graph_gexf_skips_missing_unit_edges_deterministically(tmp_path):
    path = tmp_path / "graph.gexf"
    stats = export_graph_gexf(
        [unit("unit-a", "Alpha")],
        [
            edge("edge-z", "unit-z", "unit-a"),
            edge("edge-b", "unit-a", "unit-b"),
        ],
        path,
    )

    assert stats == {
        "path": str(path),
        "units_exported": 1,
        "edges_exported": 0,
        "skipped_edges": [
            {
                "id": "edge-b",
                "from_unit_id": "unit-a",
                "to_unit_id": "unit-b",
                "reason": "missing_units:unit-b",
            },
            {
                "id": "edge-z",
                "from_unit_id": "unit-z",
                "to_unit_id": "unit-a",
                "reason": "missing_units:unit-z",
            },
        ],
    }
    exported = nx.read_gexf(path)
    assert set(exported.nodes) == {"unit-a"}
    assert exported.number_of_edges() == 0
