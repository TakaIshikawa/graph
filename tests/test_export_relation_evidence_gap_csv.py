from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_relation_evidence_gap_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    relation: EdgeRelation | str = EdgeRelation.REFERENCES,
    source: EdgeSource | str | None = EdgeSource.SOURCE,
    *,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=source,
        weight=1.0,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_evidence_gap_csv_includes_edges_with_no_sources():
    text = export_relation_evidence_gap_csv([edge("a", source=None)])

    assert rows(text) == [
        {
            "edge_id": "a",
            "relation": "references",
            "from_unit_id": "from-a",
            "to_unit_id": "to-a",
            "source_count": "0",
            "confidence_summary": "",
            "gap_reason": "no_sources",
        }
    ]


def test_relation_evidence_gap_csv_includes_low_confidence_edges():
    text = export_relation_evidence_gap_csv(
        [
            edge(
                "a",
                source=EdgeSource.SOURCE,
                metadata={"source_id": "doc-1", "confidence": 0.25},
            )
        ]
    )

    assert rows(text)[0] == {
        "edge_id": "a",
        "relation": "references",
        "from_unit_id": "from-a",
        "to_unit_id": "to-a",
        "source_count": "1",
        "confidence_summary": "0.25",
        "gap_reason": "low_confidence",
    }


def test_relation_evidence_gap_csv_excludes_complete_evidence():
    text = export_relation_evidence_gap_csv(
        [
            edge("complete", metadata={"source_id": "doc-1", "confidence": 0.9}),
            edge("missing-meta", metadata={}),
        ]
    )

    assert rows(text) == [
        {
            "edge_id": "missing-meta",
            "relation": "references",
            "from_unit_id": "from-missing-meta",
            "to_unit_id": "to-missing-meta",
            "source_count": "1",
            "confidence_summary": "",
            "gap_reason": "missing_source_metadata",
        }
    ]


def test_relation_evidence_gap_csv_sorts_deterministically_and_validates_threshold():
    edges = [
        edge("b", relation=EdgeRelation.RELATES_TO, source=None),
        edge("a", relation=EdgeRelation.BUILDS_ON, metadata={}),
        edge("c", relation=EdgeRelation.REFERENCES, metadata={"source_id": "doc-1", "confidence": 0.1}),
    ]

    assert export_relation_evidence_gap_csv(edges) == export_relation_evidence_gap_csv(reversed(edges))
    assert [row["gap_reason"] for row in rows(export_relation_evidence_gap_csv(edges))] == [
        "low_confidence",
        "missing_source_metadata",
        "no_sources",
    ]

    with pytest.raises(ValueError, match="low_threshold"):
        export_relation_evidence_gap_csv([], low_threshold=True)


def test_relation_evidence_gap_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "gaps.csv"
    edges = [edge("a", source=None)]

    expected = export_relation_evidence_gap_csv(edges)
    stats = export_relation_evidence_gap_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "rows_exported": 1,
        "low_threshold": 0.5,
        "bytes_written": path.stat().st_size,
    }
