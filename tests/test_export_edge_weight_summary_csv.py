from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.edge_weight_summary_csv import export_edge_weight_summary_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    relation: EdgeRelation | str | None,
    source: EdgeSource | str | None,
    weight: object,
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


def test_edge_weight_summary_groups_by_relation_and_source():
    text = export_edge_weight_summary_csv(
        [
            edge("a", EdgeRelation.REFERENCES, EdgeSource.SOURCE, 0.1),
            edge("b", EdgeRelation.REFERENCES, EdgeSource.SOURCE, 0.75),
            edge("c", EdgeRelation.REFERENCES, EdgeSource.SOURCE, 1.0),
            edge("d", EdgeRelation.RELATES_TO, EdgeSource.MANUAL, 0.5),
        ]
    )

    assert text.splitlines()[0] == (
        "relation,source,edge_count,min_weight,max_weight,average_weight,"
        "weak_edge_count,strong_edge_count"
    )
    assert rows(text) == [
        {
            "relation": "references",
            "source": "source",
            "edge_count": "3",
            "min_weight": "0.10",
            "max_weight": "1.00",
            "average_weight": "0.62",
            "weak_edge_count": "1",
            "strong_edge_count": "2",
        },
        {
            "relation": "relates_to",
            "source": "manual",
            "edge_count": "1",
            "min_weight": "0.50",
            "max_weight": "0.50",
            "average_weight": "0.50",
            "weak_edge_count": "0",
            "strong_edge_count": "0",
        },
    ]


def test_edge_weight_summary_normalizes_blank_relation_and_source():
    text = export_edge_weight_summary_csv(
        [
            edge("a", None, None, 0.2),
            edge("b", "  Cites\nStrongly  ", " Import\nRule ", 0.9),
            edge("c", " \n\t ", " \t ", "bad weight"),
        ]
    )

    assert rows(text) == [
        {
            "relation": "Cites Strongly",
            "source": "Import Rule",
            "edge_count": "1",
            "min_weight": "0.90",
            "max_weight": "0.90",
            "average_weight": "0.90",
            "weak_edge_count": "0",
            "strong_edge_count": "1",
        },
        {
            "relation": "Unknown",
            "source": "Unknown",
            "edge_count": "2",
            "min_weight": "0.00",
            "max_weight": "0.20",
            "average_weight": "0.10",
            "weak_edge_count": "2",
            "strong_edge_count": "0",
        },
    ]


def test_edge_weight_summary_thresholds_affect_counts_and_validate():
    text = export_edge_weight_summary_csv(
        [
            edge("a", "Relation", "Source", 0.4),
            edge("b", "Relation", "Source", 0.7),
            edge("c", "Relation", "Source", 0.9),
        ],
        weak_threshold=0.75,
        strong_threshold=0.9,
    )

    assert rows(text)[0]["weak_edge_count"] == "2"
    assert rows(text)[0]["strong_edge_count"] == "1"

    invalid_calls = [
        {"weak_threshold": -0.1},
        {"strong_threshold": 1.1},
        {"weak_threshold": 0.8, "strong_threshold": 0.8},
        {"weak_threshold": True},
        {"strong_threshold": "0.7"},
    ]
    for kwargs in invalid_calls:
        with pytest.raises(ValueError):
            export_edge_weight_summary_csv([], **kwargs)


def test_edge_weight_summary_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "edge-weights.csv"
    edges = [edge("a", "Relation", "Source", 0.6)]

    expected = export_edge_weight_summary_csv(edges)
    stats = export_edge_weight_summary_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "group_count": 1,
        "rows_exported": 1,
        "weak_threshold": 0.25,
        "strong_threshold": 0.75,
        "bytes_written": path.stat().st_size,
    }


def test_edge_weight_summary_is_deterministic_for_reversed_input():
    edges = [
        edge("a", "Relation B", "Source", 0.7),
        edge("b", "Relation A", "Source B", 0.4),
        edge("c", "Relation A", "Source A", 0.9),
    ]

    assert export_edge_weight_summary_csv(edges) == export_edge_weight_summary_csv(reversed(edges))
