from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.export import export_graph_cytoscape
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


def decode(text: str) -> dict:
    return json.loads(text)


def test_export_graph_cytoscape_returns_elements_shape_without_content_by_default():
    graph = decode(
        export_graph_cytoscape(
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
        )
    )

    assert graph["elements"]["nodes"][0] == {
        "data": {
            "id": "unit-a",
            "label": "Alpha",
            "source_project": "max",
            "source_id": "source-unit-a",
            "source_entity_type": "insight",
            "content_type": "insight",
            "metadata": {"project": "max", "reviewed_at": "2026-05-01T10:15:00+00:00"},
            "tags": ["solar", "storage"],
        }
    }
    assert "content" not in graph["elements"]["nodes"][0]["data"]
    assert graph["elements"]["edges"] == [
        {
            "data": {
                "id": "edge-a",
                "source": "unit-a",
                "target": "unit-b",
                "label": "builds_on",
                "relation": "builds_on",
                "weight": 0.75,
                "edge_source": "manual",
                "metadata": {"notes": ["curated"], "observed_at": "2026-05-01T12:30:00+00:00"},
            }
        }
    ]


def test_export_graph_cytoscape_is_deterministic_for_input_order():
    units_a = [unit("unit-b", "Beta", "Beta content"), unit("unit-a", "Alpha", "Alpha content")]
    units_b = list(reversed(units_a))
    edges_a = [
        edge("edge-c", "unit-b", "unit-a", EdgeRelation.RELATES_TO),
        edge("edge-b", "unit-a", "unit-b", EdgeRelation.REFERENCES),
        edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
    ]
    edges_b = list(reversed(edges_a))

    text_a = export_graph_cytoscape(units_a, edges_a)
    text_b = export_graph_cytoscape(units_b, edges_b)
    graph = decode(text_a)

    assert text_a == text_b
    assert [node["data"]["id"] for node in graph["elements"]["nodes"]] == ["unit-a", "unit-b"]
    assert [item["data"]["id"] for item in graph["elements"]["edges"]] == ["edge-a", "edge-b", "edge-c"]
    assert text_a.endswith("\n")


def test_export_graph_cytoscape_include_content_adds_node_content():
    graph = decode(export_graph_cytoscape([unit("unit-a", "Alpha", "Alpha content")], [], include_content=True))

    assert graph["elements"]["nodes"][0]["data"]["content"] == "Alpha content"


def test_export_graph_cytoscape_writes_file_and_counts_skipped_edges(tmp_path):
    path = tmp_path / "nested" / "cytoscape.json"
    stats = export_graph_cytoscape(
        [unit("unit-a", "Alpha", "Alpha content"), unit("unit-b", "Beta", "Beta content")],
        [
            edge("edge-a", "unit-a", "unit-b"),
            edge("edge-missing-source", "unit-missing", "unit-b"),
            edge("edge-missing-target", "unit-a", "unit-missing"),
        ],
        path,
    )

    text = path.read_text(encoding="utf-8")
    graph = decode(text)

    assert stats == {
        "path": str(path),
        "node_count": 2,
        "edge_count": 1,
        "skipped_edge_count": 2,
        "bytes_written": len(text.encode("utf-8")),
    }
    assert [item["data"]["id"] for item in graph["elements"]["edges"]] == ["edge-a"]
