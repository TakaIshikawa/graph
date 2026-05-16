from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_reciprocity_gaps_csv
from graph.export.relation_reciprocity_gaps_csv import DEFAULT_RECIPROCAL_MAP
from graph.types.enums import EdgeRelation
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    source: str,
    target: str,
    relation: EdgeRelation | str,
    *,
    weight: object = 1.0,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=relation,
        weight=weight,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_reciprocity_gaps_csv_empty_input_has_header_only():
    assert export_relation_reciprocity_gaps_csv([]) == (
        "source_id,target_id,relation_type,weight,reciprocal_relation_type,reciprocal_present,"
        "reciprocal_weight,gap_reason,evidence_count\n"
    )


def test_relation_reciprocity_gaps_csv_reports_missing_reverse_relation():
    text = export_relation_reciprocity_gaps_csv(
        [
            edge("a", "u2", "u3", "references", weight=0.5, metadata={"evidence": ["doc"]}),
            edge("b", "u1", "u2", "supports", weight=2),
        ]
    )

    assert rows(text) == [
        {
            "source_id": "u1",
            "target_id": "u2",
            "relation_type": "supports",
            "weight": "2.00",
            "reciprocal_relation_type": "supported_by",
            "reciprocal_present": "false",
            "reciprocal_weight": "",
            "gap_reason": "missing_reverse_edge",
            "evidence_count": "0",
        },
        {
            "source_id": "u2",
            "target_id": "u3",
            "relation_type": "references",
            "weight": "0.50",
            "reciprocal_relation_type": "referenced_by",
            "reciprocal_present": "false",
            "reciprocal_weight": "",
            "gap_reason": "missing_reverse_edge",
            "evidence_count": "1",
        },
    ]


def test_relation_reciprocity_gaps_csv_omits_edges_with_expected_reverse():
    text = export_relation_reciprocity_gaps_csv(
        [
            edge("a", "u1", "u2", "references"),
            edge("b", "u2", "u1", "referenced_by"),
            edge("c", "u3", "u4", EdgeRelation.RELATES_TO),
            edge("d", "u4", "u3", EdgeRelation.RELATES_TO),
        ]
    )

    assert rows(text) == []


def test_relation_reciprocity_gaps_csv_supports_mappings_and_custom_map_without_mutating_defaults():
    text = export_relation_reciprocity_gaps_csv(
        [
            {"source_id": "a", "target_id": "b", "relation_type": "precedes", "weight": "heavy"},
            {"source_id": "b", "target_id": "a", "relation_type": "follows", "weight": 1},
            {"source_id": "x", "target_id": "y", "relation_type": "precedes", "metadata": {"evidence_count": 3}},
        ],
        reciprocal_map={"precedes": "follows", "follows": "precedes"},
    )

    assert DEFAULT_RECIPROCAL_MAP.get("precedes") is None
    assert rows(text) == [
        {
            "source_id": "x",
            "target_id": "y",
            "relation_type": "precedes",
            "weight": "0.00",
            "reciprocal_relation_type": "follows",
            "reciprocal_present": "false",
            "reciprocal_weight": "",
            "gap_reason": "missing_reverse_edge",
            "evidence_count": "3",
        }
    ]


def test_relation_reciprocity_gaps_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "relation-reciprocity-gaps.csv"
    edges = [edge("a", "u1", "u2", "references")]

    expected = export_relation_reciprocity_gaps_csv(edges)
    stats = export_relation_reciprocity_gaps_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "gap_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
