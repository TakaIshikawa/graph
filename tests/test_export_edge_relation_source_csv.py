from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_edge_relation_source_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    relation: EdgeRelation | str,
    source: EdgeSource | str,
    *,
    weight: float = 1.0,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=source,
        weight=weight,
        metadata={},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_relation_source_csv_aggregates_counts_weights_and_percentages():
    text = export_edge_relation_source_csv(
        [
            edge("a", EdgeRelation.REFERENCES, EdgeSource.SOURCE, weight=2.0),
            edge("b", EdgeRelation.REFERENCES, EdgeSource.SOURCE, weight=1.0),
            edge("c", EdgeRelation.REFERENCES, EdgeSource.MANUAL, weight=3.0),
            edge("d", EdgeRelation.BUILDS_ON, EdgeSource.INFERRED, weight=0.5),
        ]
    )

    assert rows(text) == [
        {
            "relation": "builds_on",
            "edge_source": "inferred",
            "edge_count": "1",
            "total_weight": "0.50",
            "average_weight": "0.50",
            "percent_of_relation": "100.00",
        },
        {
            "relation": "references",
            "edge_source": "manual",
            "edge_count": "1",
            "total_weight": "3.00",
            "average_weight": "3.00",
            "percent_of_relation": "33.33",
        },
        {
            "relation": "references",
            "edge_source": "source",
            "edge_count": "2",
            "total_weight": "3.00",
            "average_weight": "1.50",
            "percent_of_relation": "66.67",
        },
    ]


def test_edge_relation_source_csv_normalizes_string_values_and_empty_edges():
    assert export_edge_relation_source_csv([]) == (
        "relation,edge_source,edge_count,total_weight,average_weight,percent_of_relation\n"
    )

    text = export_edge_relation_source_csv(
        [edge("a", "custom relation", "custom source", weight=2.25)]
    )

    assert rows(text)[0] == {
        "relation": "custom relation",
        "edge_source": "custom source",
        "edge_count": "1",
        "total_weight": "2.25",
        "average_weight": "2.25",
        "percent_of_relation": "100.00",
    }


def test_edge_relation_source_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "edge-source.csv"
    edges = [
        edge("a", EdgeRelation.RELATES_TO, EdgeSource.INFERRED),
        edge("b", EdgeRelation.RELATES_TO, EdgeSource.MANUAL),
    ]

    expected = export_edge_relation_source_csv(edges)
    stats = export_edge_relation_source_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 2,
        "rows_exported": 2,
        "relation_count": 1,
        "bytes_written": path.stat().st_size,
    }


def test_edge_relation_source_csv_is_deterministic_for_reversed_input():
    edges = [
        edge("a", EdgeRelation.RELATES_TO, EdgeSource.MANUAL),
        edge("b", EdgeRelation.RELATES_TO, EdgeSource.INFERRED),
        edge("c", EdgeRelation.REFERENCES, EdgeSource.SOURCE),
    ]

    assert export_edge_relation_source_csv(edges) == export_edge_relation_source_csv(reversed(edges))
