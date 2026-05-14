from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_evidence_density_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    relation: EdgeRelation | str | None = EdgeRelation.REFERENCES,
    *,
    source: EdgeSource | str | None = EdgeSource.SOURCE,
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


def test_relation_evidence_density_empty_input_returns_header_only_csv():
    assert export_relation_evidence_density_csv([]) == (
        "relation,edge_count,edges_with_source,average_confidence,average_weight,"
        "average_metadata_keys,evidence_density_score\n"
    )


def test_relation_evidence_density_groups_by_relation():
    text = export_relation_evidence_density_csv(
        [
            edge("a", metadata={"confidence": 0.5, "source_id": "doc-1"}),
            edge("b", metadata={"confidence": 1.0}),
            edge("c", EdgeRelation.RELATES_TO, source=None, weight=2.0, metadata={}),
        ]
    )

    assert rows(text) == [
        {
            "relation": "references",
            "edge_count": "2",
            "edges_with_source": "2",
            "average_confidence": "0.75",
            "average_weight": "1.00",
            "average_metadata_keys": "1.50",
            "evidence_density_score": "1.00",
        },
        {
            "relation": "relates_to",
            "edge_count": "1",
            "edges_with_source": "0",
            "average_confidence": "",
            "average_weight": "2.00",
            "average_metadata_keys": "0.00",
            "evidence_density_score": "0.25",
        },
    ]


def test_relation_evidence_density_unknown_relation_and_invalid_numbers():
    text = export_relation_evidence_density_csv(
        [edge("a", relation=None, source=None, weight="heavy", metadata={"confidence": "high", "": "skip"})]
    )

    assert rows(text) == [
        {
            "relation": "Unknown",
            "edge_count": "1",
            "edges_with_source": "0",
            "average_confidence": "",
            "average_weight": "",
            "average_metadata_keys": "1.00",
            "evidence_density_score": "0.25",
        }
    ]


def test_relation_evidence_density_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "evidence-density.csv"
    edges = [edge("a")]

    expected = export_relation_evidence_density_csv(edges)
    stats = export_relation_evidence_density_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "relation_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }


def test_relation_evidence_density_is_deterministic_for_reversed_input():
    edges = [
        edge("a", EdgeRelation.RELATES_TO),
        edge("b", EdgeRelation.REFERENCES),
        edge("c", EdgeRelation.REFERENCES, metadata={"confidence": 0.2}),
    ]

    assert export_relation_evidence_density_csv(edges) == export_relation_evidence_density_csv(reversed(edges))
