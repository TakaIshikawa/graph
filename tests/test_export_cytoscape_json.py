from __future__ import annotations

import copy
import json
from datetime import datetime, timezone

from graph.export import export_graph_cytoscape_json
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
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
        content_type=ContentType.FINDING,
        metadata=metadata or {},
        tags=["graph"],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    *,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.MANUAL,
        weight=0.75,
        metadata=metadata or {},
        created_at=EDGE_TIME,
    )


def test_export_graph_cytoscape_json_returns_node_only_elements():
    graph = export_graph_cytoscape_json([unit("unit-a", "Alpha")], [])

    assert graph == {
        "elements": {
            "nodes": [
                {
                    "data": {
                        "id": "unit-a",
                        "label": "Alpha",
                        "type": "insight",
                        "source_project": "max",
                        "source_id": "source-unit-a",
                        "source_entity_type": "insight",
                        "content_type": "finding",
                        "tags": ["graph"],
                        "metadata": {},
                    }
                }
            ],
            "edges": [],
        }
    }
    json.dumps(graph)


def test_export_graph_cytoscape_json_includes_edge_data_and_writes_file(tmp_path):
    path = tmp_path / "nested" / "cytoscape.json"

    stats = export_graph_cytoscape_json(
        [unit("unit-b", "Beta"), unit("unit-a", "Alpha")],
        [edge("edge-a", "unit-a", "unit-b", relation=EdgeRelation.BUILDS_ON)],
        path,
    )
    graph = json.loads(path.read_text(encoding="utf-8"))

    assert [node["data"]["id"] for node in graph["elements"]["nodes"]] == ["unit-a", "unit-b"]
    assert graph["elements"]["edges"] == [
        {
            "data": {
                "id": "edge-a",
                "source": "unit-a",
                "target": "unit-b",
                "label": "builds_on",
                "type": "builds_on",
                "relation": "builds_on",
                "weight": 0.75,
                "edge_source": "manual",
                "metadata": {},
            }
        }
    ]
    assert stats == {
        "path": str(path),
        "node_count": 2,
        "edge_count": 1,
        "bytes_written": path.stat().st_size,
    }


def test_export_graph_cytoscape_json_preserves_metadata_without_mutating_sources():
    node_metadata = {"published_at": UNIT_TIME, "project": SourceProject.MAX, "nested": {"rank": 3}}
    edge_metadata = {"observed_at": EDGE_TIME, "notes": ["curated"]}
    alpha = unit("unit-a", "Alpha", metadata=node_metadata)
    relation = edge("edge-a", "unit-a", "unit-b", metadata=edge_metadata)
    original_unit_metadata = copy.deepcopy(alpha.metadata)
    original_edge_metadata = copy.deepcopy(relation.metadata)

    graph = export_graph_cytoscape_json([alpha, unit("unit-b", "Beta")], [relation])

    node_data = graph["elements"]["nodes"][0]["data"]
    edge_data = graph["elements"]["edges"][0]["data"]
    assert node_data["metadata"] == {
        "nested": {"rank": 3},
        "project": "max",
        "published_at": "2026-05-01T10:15:00+00:00",
    }
    assert edge_data["metadata"] == {
        "notes": ["curated"],
        "observed_at": "2026-05-01T12:30:00+00:00",
    }
    assert alpha.metadata == original_unit_metadata
    assert relation.metadata == original_edge_metadata
