from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_endpoint_type_matrix_csv
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, entity_type: str | None, source_project: SourceProject | str = SourceProject.MAX):
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type=entity_type,
        title=f"Title {unit_id}",
        content="content",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation | str | None = EdgeRelation.REFERENCES,
    *,
    weight: object = 1.0,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
        source=EdgeSource.INFERRED,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_endpoint_type_matrix_empty_edges_returns_header_only_csv():
    assert export_relation_endpoint_type_matrix_csv([], []) == (
        "source_entity_type,relation,target_entity_type,edge_count,average_weight,average_confidence\n"
    )


def test_relation_endpoint_type_matrix_groups_endpoint_patterns():
    units = [unit("a", "note"), unit("b", "bookmark"), unit("c", "note")]
    edges = [
        edge("e1", "a", "b", EdgeRelation.REFERENCES, weight=2.0, metadata={"confidence": 0.5}),
        edge("e2", "c", "b", EdgeRelation.REFERENCES, weight=4.0, metadata={"confidence": 1.0}),
        edge("e3", "b", "a", EdgeRelation.RELATES_TO, weight=1.0),
    ]

    assert rows(export_relation_endpoint_type_matrix_csv(units, edges)) == [
        {
            "source_entity_type": "bookmark",
            "relation": "relates_to",
            "target_entity_type": "note",
            "edge_count": "1",
            "average_weight": "1.00",
            "average_confidence": "",
        },
        {
            "source_entity_type": "note",
            "relation": "references",
            "target_entity_type": "bookmark",
            "edge_count": "2",
            "average_weight": "3.00",
            "average_confidence": "0.75",
        },
    ]


def test_relation_endpoint_type_matrix_uses_unknown_fallbacks_and_ignores_invalid_numbers():
    units = [unit("a", None)]
    text = export_relation_endpoint_type_matrix_csv(
        units,
        [edge("e1", "a", "missing", None, weight="heavy", metadata={"confidence": "high"})],
    )

    assert rows(text) == [
        {
            "source_entity_type": "Unknown",
            "relation": "Unknown",
            "target_entity_type": "Unknown",
            "edge_count": "1",
            "average_weight": "",
            "average_confidence": "",
        }
    ]


def test_relation_endpoint_type_matrix_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "endpoint-matrix.csv"
    units = [unit("a", "note"), unit("b", "bookmark")]
    edges = [edge("e1", "a", "b")]

    expected = export_relation_endpoint_type_matrix_csv(units, edges)
    stats = export_relation_endpoint_type_matrix_csv(units, edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "edge_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_relation_endpoint_type_matrix_is_deterministic_for_reversed_input():
    units = [unit("a", "note"), unit("b", "bookmark"), unit("c", "task")]
    edges = [edge("e1", "a", "b"), edge("e2", "c", "a", EdgeRelation.BUILDS_ON)]

    assert export_relation_endpoint_type_matrix_csv(units, edges) == export_relation_endpoint_type_matrix_csv(
        reversed(units), reversed(edges)
    )
