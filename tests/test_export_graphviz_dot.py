from __future__ import annotations

from datetime import datetime, timezone

from graph.export import export_graphviz_dot
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
        created_at=UNIT_TIME,
    )


def test_export_graphviz_dot_outputs_directed_graph():
    text = export_graphviz_dot(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON)],
    )

    assert text == (
        'digraph "KnowledgeGraph" {\n'
        '  "unit-a" [label="Alpha"];\n'
        '  "unit-b" [label="Beta"];\n'
        '  "unit-a" -> "unit-b" [label="builds_on"];\n'
        "}\n"
    )


def test_export_graphviz_dot_escapes_node_ids_node_labels_and_edge_labels():
    text = export_graphviz_dot(
        [unit('unit "a"\\1', 'Title "quoted"\\line\nnext'), unit("unit-b", "Beta")],
        [edge("edge-a", 'unit "a"\\1', "unit-b", EdgeRelation.REFERENCES)],
        graph_name='Graph "Name"\nNext',
    )

    assert text.startswith('digraph "Graph \\"Name\\"\\nNext" {\n')
    assert '  "unit \\"a\\"\\\\1" [label="Title \\"quoted\\"\\\\line\\nnext"];' in text
    assert '  "unit \\"a\\"\\\\1" -> "unit-b" [label="references"];' in text


def test_export_graphviz_dot_can_omit_edge_labels():
    text = export_graphviz_dot(
        [unit("unit-a", "Alpha"), unit("unit-b", "Beta")],
        [edge("edge-a", "unit-a", "unit-b", EdgeRelation.RELATES_TO)],
        include_edge_labels=False,
    )

    assert '  "unit-a" -> "unit-b";' in text
    assert "[label=\"relates_to\"]" not in text


def test_export_graphviz_dot_sorts_output_deterministically():
    units = [unit("unit-b", "Beta"), unit("unit-a", "Alpha")]
    edges = [
        edge("edge-c", "unit-b", "unit-a", EdgeRelation.RELATES_TO),
        edge("edge-b", "unit-a", "unit-b", EdgeRelation.REFERENCES),
        edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
    ]

    first = export_graphviz_dot(units, edges)
    second = export_graphviz_dot(reversed(units), reversed(edges))

    assert first == second
    assert first.splitlines() == [
        'digraph "KnowledgeGraph" {',
        '  "unit-a" [label="Alpha"];',
        '  "unit-b" [label="Beta"];',
        '  "unit-a" -> "unit-b" [label="builds_on"];',
        '  "unit-a" -> "unit-b" [label="references"];',
        '  "unit-b" -> "unit-a" [label="relates_to"];',
        "}",
    ]


def test_export_graphviz_dot_is_available_from_graph_export():
    assert callable(export_graphviz_dot)
