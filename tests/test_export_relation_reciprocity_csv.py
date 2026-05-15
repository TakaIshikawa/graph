from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_reciprocity_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.RELATES_TO,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.INFERRED,
        weight=1.0,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_reciprocity_csv_empty_input_returns_header():
    assert export_relation_reciprocity_csv([]) == (
        "from_unit_id,to_unit_id,forward_relations,reverse_relations,forward_edge_count,"
        "reverse_edge_count,reciprocity_status\n"
    )


def test_relation_reciprocity_csv_groups_reciprocal_edges_for_unordered_pair():
    text = export_relation_reciprocity_csv(
        [
            edge("a", "unit-b", "unit-a", relation=EdgeRelation.REFERENCES),
            edge("b", "unit-a", "unit-b", relation=EdgeRelation.RELATES_TO),
            edge("c", "unit-a", "unit-b", relation=EdgeRelation.RELATES_TO),
        ]
    )

    assert rows(text) == [
        {
            "from_unit_id": "unit-a",
            "to_unit_id": "unit-b",
            "forward_relations": "relates_to (2)",
            "reverse_relations": "references",
            "forward_edge_count": "2",
            "reverse_edge_count": "1",
            "reciprocity_status": "reciprocal",
        }
    ]


def test_relation_reciprocity_csv_includes_one_way_pairs_in_either_direction():
    text = export_relation_reciprocity_csv(
        [
            edge("a", "unit-a", "unit-c", relation=EdgeRelation.BUILDS_ON),
            edge("b", "unit-z", "unit-y", relation=EdgeRelation.CHALLENGES),
        ]
    )

    assert rows(text) == [
        {
            "from_unit_id": "unit-a",
            "to_unit_id": "unit-c",
            "forward_relations": "builds_on",
            "reverse_relations": "",
            "forward_edge_count": "1",
            "reverse_edge_count": "0",
            "reciprocity_status": "one_way",
        },
        {
            "from_unit_id": "unit-y",
            "to_unit_id": "unit-z",
            "forward_relations": "",
            "reverse_relations": "challenges",
            "forward_edge_count": "0",
            "reverse_edge_count": "1",
            "reciprocity_status": "one_way",
        },
    ]


def test_relation_reciprocity_csv_sorts_by_status_and_unit_ids():
    edges = [
        edge("a", "unit-c", "unit-d"),
        edge("b", "unit-b", "unit-a"),
        edge("c", "unit-a", "unit-b"),
    ]

    assert export_relation_reciprocity_csv(edges) == export_relation_reciprocity_csv(reversed(edges))
    assert [row["reciprocity_status"] for row in rows(export_relation_reciprocity_csv(edges))] == [
        "one_way",
        "reciprocal",
    ]


def test_relation_reciprocity_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reciprocity.csv"
    edges = [edge("a", "unit-a", "unit-b")]

    expected = export_relation_reciprocity_csv(edges)
    stats = export_relation_reciprocity_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "pair_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
