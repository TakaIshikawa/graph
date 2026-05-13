from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.relation_confidence_summary_csv import export_relation_confidence_summary_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    *,
    relation: EdgeRelation | str | None = EdgeRelation.RELATES_TO,
    source: EdgeSource | str | None = EdgeSource.INFERRED,
    weight: object = 1.0,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=source,
        weight=weight,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_confidence_summary_empty_input_has_header_only():
    assert export_relation_confidence_summary_csv([]) == (
        "relation,source,source_project,edge_count,confidence_count,missing_confidence_count,"
        "min_confidence,max_confidence,average_confidence,low_confidence_count,"
        "medium_confidence_count,high_confidence_count\n"
    )


def test_relation_confidence_summary_groups_by_relation_source_and_project():
    text = export_relation_confidence_summary_csv(
        [
            edge(
                "a",
                relation=EdgeRelation.REFERENCES,
                source=EdgeSource.SOURCE,
                metadata={"source_project": "alpha", "confidence": 0.2},
            ),
            edge(
                "b",
                relation=EdgeRelation.REFERENCES,
                source=EdgeSource.SOURCE,
                metadata={"source_project": "alpha", "score": 0.7},
            ),
            edge(
                "c",
                relation=EdgeRelation.REFERENCES,
                source=EdgeSource.MANUAL,
                metadata={"project": "beta", "confidence": 0.9},
            ),
            edge("d", relation="custom", source=None, weight=0.8),
        ],
        low_threshold=0.5,
        high_threshold=0.8,
    )

    assert rows(text) == [
        {
            "relation": "custom",
            "source": "Unknown",
            "source_project": "Unknown",
            "edge_count": "1",
            "confidence_count": "1",
            "missing_confidence_count": "0",
            "min_confidence": "0.80",
            "max_confidence": "0.80",
            "average_confidence": "0.80",
            "low_confidence_count": "0",
            "medium_confidence_count": "0",
            "high_confidence_count": "1",
        },
        {
            "relation": "references",
            "source": "manual",
            "source_project": "beta",
            "edge_count": "1",
            "confidence_count": "1",
            "missing_confidence_count": "0",
            "min_confidence": "0.90",
            "max_confidence": "0.90",
            "average_confidence": "0.90",
            "low_confidence_count": "0",
            "medium_confidence_count": "0",
            "high_confidence_count": "1",
        },
        {
            "relation": "references",
            "source": "source",
            "source_project": "alpha",
            "edge_count": "2",
            "confidence_count": "2",
            "missing_confidence_count": "0",
            "min_confidence": "0.20",
            "max_confidence": "0.70",
            "average_confidence": "0.45",
            "low_confidence_count": "1",
            "medium_confidence_count": "1",
            "high_confidence_count": "0",
        },
    ]


def test_relation_confidence_summary_falls_back_to_weight_only_without_explicit_keys():
    text = export_relation_confidence_summary_csv(
        [
            edge("a", weight=0.9),
            edge("b", weight=0.9, metadata={"score": "bad"}),
            edge("c", weight=0.9, metadata={"confidence": None}),
        ]
    )

    row = rows(text)[0]
    assert row["edge_count"] == "3"
    assert row["confidence_count"] == "1"
    assert row["missing_confidence_count"] == "2"
    assert row["average_confidence"] == "0.90"


def test_relation_confidence_summary_thresholds_affect_buckets_and_validate():
    text = export_relation_confidence_summary_csv(
        [edge("a", weight=0.4), edge("b", weight=0.7), edge("c", weight=0.9)],
        low_threshold=0.75,
        high_threshold=0.9,
    )

    assert rows(text)[0]["low_confidence_count"] == "2"
    assert rows(text)[0]["medium_confidence_count"] == "0"
    assert rows(text)[0]["high_confidence_count"] == "1"

    invalid_calls = [
        {"low_threshold": -0.1},
        {"high_threshold": 1.1},
        {"low_threshold": 0.8, "high_threshold": 0.8},
        {"low_threshold": True},
        {"high_threshold": "0.7"},
    ]
    for kwargs in invalid_calls:
        with pytest.raises(ValueError):
            export_relation_confidence_summary_csv([], **kwargs)


def test_relation_confidence_summary_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "relation-confidence.csv"
    edges = [edge("a", weight=0.6)]

    expected = export_relation_confidence_summary_csv(edges)
    stats = export_relation_confidence_summary_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "group_count": 1,
        "rows_exported": 1,
        "low_threshold": 0.5,
        "high_threshold": 0.8,
        "bytes_written": path.stat().st_size,
    }


def test_relation_confidence_summary_is_deterministic_for_reversed_input():
    edges = [
        edge("a", relation="Relation B", source="Source", metadata={"source_project": "Project", "score": 0.7}),
        edge("b", relation="Relation A", source="Source B", metadata={"source_project": "Project", "score": 0.4}),
        edge("c", relation="Relation A", source="Source A", metadata={"source_project": "Project", "score": 0.9}),
    ]

    assert export_relation_confidence_summary_csv(edges) == export_relation_confidence_summary_csv(
        reversed(edges)
    )
