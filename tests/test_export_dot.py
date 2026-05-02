from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_graph_dot
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    title: str,
    tags: list[str] | None = None,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
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
    weight: float = 1.0,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
        weight=weight,
        metadata=metadata or {},
        created_at=EDGE_TIME,
    )


def test_export_graph_dot_writes_nodes_edges_and_edge_attributes():
    dot = export_graph_dot(
        [
            unit("unit-a", "Alpha", ["solar", "storage"]),
            unit("unit-b", "Beta", ["storage"]),
        ],
        [edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON, weight=0.75)],
    )

    assert dot == (
        'digraph "KnowledgeGraph" {\n'
        '  "unit-a" [label="Alpha", title="Alpha", source_project="max", tags="[\\"solar\\",\\"storage\\"]"];\n'
        '  "unit-b" [label="Beta", title="Beta", source_project="max", tags="[\\"storage\\"]"];\n'
        '  "unit-a" -> "unit-b" [label="builds_on", relation="builds_on", weight="0.75"];\n'
        "}\n"
    )


def test_export_graph_dot_escapes_ids_labels_and_json_attribute_values():
    dot = export_graph_dot(
        [
            unit(
                'unit "a"\\1',
                'Title "quoted"\\line\nnext',
                ['tag "one"', "slash\\tag", "line\ntag"],
            ),
            unit("unit-b", "Beta"),
        ],
        [edge("edge-a", 'unit "a"\\1', "unit-b", EdgeRelation.REFERENCES)],
        graph_name='Graph "Name"\nNext',
    )

    assert dot.startswith('digraph "Graph \\"Name\\"\\nNext" {\n')
    assert (
        '  "unit \\"a\\"\\\\1" [label="Title \\"quoted\\"\\\\line\\nnext", '
        'title="Title \\"quoted\\"\\\\line\\nnext", source_project="max", '
        'tags="[\\"tag \\\\\\"one\\\\\\"\\",\\"slash\\\\\\\\tag\\",\\"line\\\\ntag\\"]"];'
    ) in dot
    assert (
        '  "unit \\"a\\"\\\\1" -> "unit-b" [label="references", relation="references", weight="1.0"];'
        in dot
    )


def test_export_graph_dot_sorts_output_deterministically():
    units = [
        unit("unit-b", "Beta"),
        unit("unit-a", "Alpha"),
    ]
    edges = [
        edge("edge-c", "unit-b", "unit-a", EdgeRelation.RELATES_TO),
        edge("edge-b", "unit-a", "unit-b", EdgeRelation.REFERENCES),
        edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
    ]

    first = export_graph_dot(units, edges)
    second = export_graph_dot(reversed(units), reversed(edges))

    assert first == second
    assert first.splitlines() == [
        'digraph "KnowledgeGraph" {',
        '  "unit-a" [label="Alpha", title="Alpha", source_project="max", tags="[]"];',
        '  "unit-b" [label="Beta", title="Beta", source_project="max", tags="[]"];',
        '  "unit-a" -> "unit-b" [label="builds_on", relation="builds_on", weight="1.0"];',
        '  "unit-a" -> "unit-b" [label="references", relation="references", weight="1.0"];',
        '  "unit-b" -> "unit-a" [label="relates_to", relation="relates_to", weight="1.0"];',
        "}",
    ]


def test_export_graph_dot_omits_and_includes_metadata_attributes():
    units = [
        unit(
            "unit-a",
            "Alpha",
            metadata={
                "project": SourceProject.MAX,
                "reviewed_at": UNIT_TIME,
                "nested": {"z": 2, "a": True},
            },
        )
    ]
    edges = [
        edge(
            "edge-a",
            "unit-a",
            "unit-b",
            metadata={"observed_at": EDGE_TIME, "notes": ["curated"]},
        )
    ]

    without_metadata = export_graph_dot(units, edges, include_metadata=False)
    with_metadata = export_graph_dot(units, edges, include_metadata=True)

    assert "metadata=" not in without_metadata
    assert (
        'metadata="{\\"nested\\":{\\"a\\":true,\\"z\\":2},'
        '\\"project\\":\\"max\\",\\"reviewed_at\\":\\"2026-05-01T10:15:00+00:00\\"}"'
    ) in with_metadata
    assert (
        'metadata="{\\"notes\\":[\\"curated\\"],\\"observed_at\\":\\"2026-05-01T12:30:00+00:00\\"}"'
    ) in with_metadata


def test_export_graph_dot_is_importable_from_graph_export():
    assert callable(export_graph_dot)
