from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.export import export_graph_adjacency_csv
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=ContentType.FINDING,
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
        metadata=metadata or {},
        created_at=EDGE_TIME,
    )


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_graph_adjacency_csv_writes_header_order_and_rows(tmp_path):
    path = tmp_path / "adjacency.csv"

    stats = export_graph_adjacency_csv(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [edge("edge-a", "unit-a", "unit-b", relation=EdgeRelation.BUILDS_ON)],
        path,
    )

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "source_id",
        "target_id",
        "edge_type",
        "edge_label",
        "source_label",
        "target_label",
    ]
    assert rows == [
        {
            "source_id": "unit-a",
            "target_id": "unit-b",
            "edge_type": "builds_on",
            "edge_label": "builds_on",
            "source_label": "Alpha",
            "target_label": "Beta",
        }
    ]
    assert stats == {
        "path": str(path),
        "nodes_scanned": 2,
        "edges_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_export_graph_adjacency_csv_empty_graph_writes_header_only(tmp_path):
    path = tmp_path / "empty.csv"

    stats = export_graph_adjacency_csv([], [], path)
    text = path.read_text(encoding="utf-8")

    assert (
        text
        == "source_id,target_id,edge_type,edge_label,source_label,target_label\n"
    )
    assert stats == {
        "path": str(path),
        "nodes_scanned": 0,
        "edges_exported": 0,
        "bytes_written": len(text.encode("utf-8")),
    }


def test_export_graph_adjacency_csv_preserves_edge_order_and_blank_missing_labels(tmp_path):
    path = tmp_path / "adjacency.csv"

    export_graph_adjacency_csv(
        [unit("unit-a", "Alpha")],
        [
            edge("edge-b", "unit-missing", "unit-a", metadata={"label": "custom label"}),
            edge("edge-a", "unit-a", "unit-missing", relation=EdgeRelation.REFERENCES),
        ],
        path,
    )

    assert read_rows(path) == [
        {
            "source_id": "unit-missing",
            "target_id": "unit-a",
            "edge_type": "relates_to",
            "edge_label": "custom label",
            "source_label": "",
            "target_label": "Alpha",
        },
        {
            "source_id": "unit-a",
            "target_id": "unit-missing",
            "edge_type": "references",
            "edge_label": "references",
            "source_label": "Alpha",
            "target_label": "",
        },
    ]
