from __future__ import annotations

import csv
from io import StringIO

from graph.export.relation_pair_summary_csv import export_relation_pair_summary_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.RELATES_TO,
    source: EdgeSource | str = EdgeSource.INFERRED,
    weight: object = 1.0,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=source,
        weight=weight,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_pair_summary_csv_empty_input_has_header_only():
    assert export_relation_pair_summary_csv([]) == (
        "from_unit_id,to_unit_id,relation_count,edge_count,sources,relations,min_weight,max_weight,"
        "average_weight,bidirectional_pair_key\n"
    )


def test_relation_pair_summary_csv_collapses_multiple_edges_for_same_ordered_pair():
    text = export_relation_pair_summary_csv(
        [
            edge("a", "u1", "u2", relation=EdgeRelation.RELATES_TO, source=EdgeSource.INFERRED, weight=1),
            edge("b", "u1", "u2", relation=EdgeRelation.RELATES_TO, source=EdgeSource.MANUAL, weight=2),
            edge("c", "u1", "u2", relation=EdgeRelation.REFERENCES, source=EdgeSource.MANUAL, weight=4),
        ]
    )

    assert rows(text) == [
        {
            "from_unit_id": "u1",
            "to_unit_id": "u2",
            "relation_count": "2",
            "edge_count": "3",
            "sources": "inferred; manual (2)",
            "relations": "references; relates_to (2)",
            "min_weight": "1.00",
            "max_weight": "4.00",
            "average_weight": "2.33",
            "bidirectional_pair_key": "u1|u2",
        }
    ]


def test_relation_pair_summary_csv_keeps_ordered_pairs_distinct_with_shared_bidirectional_key():
    text = export_relation_pair_summary_csv(
        [
            edge("a", "u2", "u1", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE, weight=0.5),
            edge("b", "u1", "u2", relation=EdgeRelation.RELATES_TO, source=EdgeSource.INFERRED, weight=1.5),
        ]
    )

    assert rows(text) == [
        {
            "from_unit_id": "u1",
            "to_unit_id": "u2",
            "relation_count": "1",
            "edge_count": "1",
            "sources": "inferred",
            "relations": "relates_to",
            "min_weight": "1.50",
            "max_weight": "1.50",
            "average_weight": "1.50",
            "bidirectional_pair_key": "u1|u2",
        },
        {
            "from_unit_id": "u2",
            "to_unit_id": "u1",
            "relation_count": "1",
            "edge_count": "1",
            "sources": "source",
            "relations": "references",
            "min_weight": "0.50",
            "max_weight": "0.50",
            "average_weight": "0.50",
            "bidirectional_pair_key": "u1|u2",
        },
    ]


def test_relation_pair_summary_csv_sorts_labels_and_formats_weights_stably():
    text = export_relation_pair_summary_csv(
        [
            edge("a", "u1", "u2", relation="zeta", source="source-z", weight="heavy"),
            edge("b", "u1", "u2", relation="alpha", source="source-a", weight=-1),
            edge("c", "u1", "u2", relation="alpha", source="source-a", weight=2),
        ]
    )

    row = rows(text)[0]
    assert row["sources"] == "source-a (2); source-z"
    assert row["relations"] == "alpha (2); zeta"
    assert row["min_weight"] == "-1.00"
    assert row["max_weight"] == "2.00"
    assert row["average_weight"] == "0.33"


def test_relation_pair_summary_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "relation-pair-summary.csv"
    edges = [edge("a", "u1", "u2")]

    expected = export_relation_pair_summary_csv(edges)
    stats = export_relation_pair_summary_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "pair_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
