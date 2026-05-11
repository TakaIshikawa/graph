from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_graph_node_edge_csv
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(unit_id: str, title: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=title,
        content=f"{title} content",
        content_type=ContentType.FINDING,
        tags=["storage", "solar", "storage"],
        metadata=metadata or {},
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def edge(edge_id: str, source: str, target: str, *, metadata: dict | None = None) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=EdgeRelation.BUILDS_ON,
        weight=2.5,
        source=EdgeSource.MANUAL,
        metadata=metadata or {},
        created_at=EDGE_TIME,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_node_edge_csv_returns_deterministic_node_and_edge_strings():
    nodes_csv, edges_csv = export_graph_node_edge_csv(
        [
            unit("unit-b", "Beta", metadata={"rank": 2}),
            unit("unit-a", "Alpha", metadata={"nested": {"b": 2, "a": 1}}),
        ],
        [edge("edge-b", "unit-b", "unit-a"), edge("edge-a", "unit-a", "unit-b")],
    )

    assert nodes_csv.splitlines()[0] == (
        "Id,Label,source_project,source_entity_type,content_type,tags,created_at,updated_at,metadata"
    )
    assert edges_csv.splitlines()[0] == "Source,Target,Type,Weight,created_at,metadata"
    assert [row["Id"] for row in rows(nodes_csv)] == ["unit-a", "unit-b"]
    assert [(row["Source"], row["Target"]) for row in rows(edges_csv)] == [
        ("unit-a", "unit-b"),
        ("unit-b", "unit-a"),
    ]
    assert rows(nodes_csv)[0]["metadata"] == '{"nested":{"a":1,"b":2}}'


def test_node_edge_csv_fields_include_metadata_and_timestamps():
    nodes_csv, edges_csv = export_graph_node_edge_csv(
        [unit("unit-a", "Alpha", metadata={"seen": True})],
        [edge("edge-a", "unit-a", "unit-b", metadata={"label": "A, quoted"})],
    )

    node_row = rows(nodes_csv)[0]
    edge_row = rows(edges_csv)[0]
    assert node_row == {
        "Id": "unit-a",
        "Label": "Alpha",
        "source_project": "max",
        "source_entity_type": "insight",
        "content_type": "finding",
        "tags": "solar;storage",
        "created_at": UNIT_TIME.isoformat(),
        "updated_at": UNIT_TIME.isoformat(),
        "metadata": '{"seen":true}',
    }
    assert edge_row == {
        "Source": "unit-a",
        "Target": "unit-b",
        "Type": "builds_on",
        "Weight": "2.5",
        "created_at": EDGE_TIME.isoformat(),
        "metadata": json.dumps({"label": "A, quoted"}, separators=(",", ":")),
    }


def test_node_edge_csv_writes_optional_paths_and_creates_parent_directories(tmp_path):
    nodes_path = tmp_path / "bundle" / "nodes.csv"
    edges_path = tmp_path / "bundle" / "edges.csv"

    nodes_csv, edges_csv = export_graph_node_edge_csv(
        [unit("unit-a", "Alpha")],
        [edge("edge-a", "unit-a", "unit-b")],
        nodes_path=nodes_path,
        edges_path=edges_path,
    )

    assert nodes_path.read_text(encoding="utf-8") == nodes_csv
    assert edges_path.read_text(encoding="utf-8") == edges_csv


def test_node_edge_csv_is_importable_from_graph_export():
    from graph.export import export_graph_node_edge_csv as imported

    assert imported is export_graph_node_edge_csv
