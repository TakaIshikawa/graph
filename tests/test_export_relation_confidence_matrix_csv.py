from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_relation_confidence_matrix_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    relation: EdgeRelation | str | None,
    confidence: object = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=EdgeSource.INFERRED,
        weight=1.0,
        metadata={} if confidence is None else {"confidence": confidence},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_confidence_matrix_csv_counts_all_buckets():
    text = export_relation_confidence_matrix_csv(
        [
            edge("a", EdgeRelation.REFERENCES, 0.2),
            edge("b", EdgeRelation.REFERENCES, 0.5),
            edge("c", EdgeRelation.REFERENCES, 0.79),
            edge("d", EdgeRelation.REFERENCES, 0.8),
            edge("e", EdgeRelation.REFERENCES, None),
            edge("f", EdgeRelation.REFERENCES, "high"),
        ]
    )

    assert rows(text) == [
        {
            "relation": "references",
            "low_count": "1",
            "medium_count": "2",
            "high_count": "1",
            "unknown_count": "2",
            "total_count": "6",
        }
    ]


def test_relation_confidence_matrix_csv_groups_missing_relation_as_unknown():
    text = export_relation_confidence_matrix_csv(
        [
            edge("a", None, 0.9),
            edge("b", "", 0.1),
            edge("c", " ", None),
        ]
    )

    assert rows(text) == [
        {
            "relation": "Unknown",
            "low_count": "1",
            "medium_count": "0",
            "high_count": "1",
            "unknown_count": "1",
            "total_count": "3",
        }
    ]


def test_relation_confidence_matrix_csv_is_deterministic_for_reversed_input():
    edges = [
        edge("a", "Zeta", 0.4),
        edge("b", "alpha", 0.6),
        edge("c", "Alpha", 0.9),
        edge("d", None, None),
    ]

    expected = rows(export_relation_confidence_matrix_csv(edges))

    assert export_relation_confidence_matrix_csv(edges) == export_relation_confidence_matrix_csv(
        reversed(edges)
    )
    assert [row["relation"] for row in expected] == ["Alpha", "alpha", "Unknown", "Zeta"]


def test_relation_confidence_matrix_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "relation-confidence.csv"
    edges = [edge("a", EdgeRelation.RELATES_TO, 0.7)]

    expected = export_relation_confidence_matrix_csv(edges)
    stats = export_relation_confidence_matrix_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "relation_count": 1,
        "rows_exported": 1,
        "low_threshold": 0.5,
        "high_threshold": 0.8,
        "bytes_written": path.stat().st_size,
    }


def test_relation_confidence_matrix_csv_thresholds_validate():
    with pytest.raises(ValueError, match="low_threshold"):
        export_relation_confidence_matrix_csv([], low_threshold=-0.1)
    with pytest.raises(ValueError, match="high_threshold"):
        export_relation_confidence_matrix_csv([], high_threshold=1.1)
    with pytest.raises(ValueError, match="less than"):
        export_relation_confidence_matrix_csv([], low_threshold=0.8, high_threshold=0.8)
    with pytest.raises(ValueError, match="low_threshold"):
        export_relation_confidence_matrix_csv([], low_threshold=True)
