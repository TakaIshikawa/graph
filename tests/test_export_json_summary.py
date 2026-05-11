from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_graph_json_summary
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 2, 12, 30, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    entity_type: str = "note",
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type=entity_type,
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags or [],
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def edge(
    edge_id: str,
    from_unit_id: str = "unit-a",
    to_unit_id: str = "unit-b",
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    *,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
        metadata=metadata or {},
        created_at=EDGE_TIME,
    )


def test_export_graph_json_summary_handles_empty_graph():
    assert json.loads(export_graph_json_summary([], [])) == {
        "edge_types": {},
        "node_types": {},
        "tag_counts": {},
        "total_edges": 0,
        "total_nodes": 0,
    }


def test_export_graph_json_summary_counts_typed_nodes_and_edges():
    summary = json.loads(
        export_graph_json_summary(
            [
                unit("unit-a", entity_type="note", tags=["solar", "storage"]),
                unit("unit-b", entity_type="paper", tags=["solar", ""]),
                unit("unit-c", entity_type="note", tags=["storage"]),
            ],
            [
                edge("edge-a", relation=EdgeRelation.RELATES_TO),
                edge("edge-b", relation=EdgeRelation.REFERENCES),
                edge("edge-c", relation=EdgeRelation.RELATES_TO),
            ],
        )
    )

    assert summary["total_nodes"] == 3
    assert summary["total_edges"] == 3
    assert summary["node_types"] == {"note": 2, "paper": 1}
    assert summary["edge_types"] == {"references": 1, "relates_to": 2}
    assert summary["tag_counts"] == {"solar": 2, "storage": 2}


def test_export_graph_json_summary_aggregates_timestamp_bounds():
    summary = json.loads(
        export_graph_json_summary(
            [
                unit("unit-a", metadata={"published_at": "2026-04-01T00:00:00+00:00"}),
                unit("unit-b", metadata={"nested": {"event_date": "2026-06-01"}}),
            ],
            [edge("edge-a", metadata={"observed_at": "2026-03-01T00:00:00+00:00"})],
        )
    )

    assert summary["min_timestamp"] == "2026-03-01T00:00:00+00:00"
    assert summary["max_timestamp"] == "2026-06-01"
