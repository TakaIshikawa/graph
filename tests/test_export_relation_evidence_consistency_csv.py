from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_evidence_consistency_csv
from graph.types.enums import EdgeRelation
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.REFERENCES,
    source: object = "manual",
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"u{edge_id}",
        to_unit_id=f"v{edge_id}",
        relation=relation,
        source=source,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_evidence_consistency_groups_and_flags_conflicts():
    text = export_relation_evidence_consistency_csv(
        [
            edge(
                "2",
                metadata={
                    "relation_type": "claim",
                    "source_project": "alpha",
                    "evidence": [{"source_id": "doc-b", "date": "2026-05-02"}],
                },
            ),
            edge(
                "1",
                metadata={
                    "relation_type": "claim",
                    "source_project": "alpha",
                    "evidence": [{"source_id": "doc-a", "date": "2026-05-01"}],
                },
            ),
            edge("3", metadata={"relation_type": "claim", "source_project": "alpha"}),
        ]
    )

    assert rows(text) == [
        {
            "relation": "references",
            "relation_type": "claim",
            "source_bucket": "alpha",
            "edge_count": "3",
            "evidence_count": "2",
            "missing_evidence_count": "1",
            "conflicting_source_count": "2",
            "conflicting_date_count": "2",
            "unit_pairs": "u1->v1; u2->v2; u3->v3",
            "sample_edge_ids": "1; 2; 3",
        }
    ]


def test_relation_evidence_consistency_handles_sources_metadata_and_dict_edges():
    text = export_relation_evidence_consistency_csv(
        [
            {
                "id": "a",
                "relation": "supports",
                "from_unit_id": "u1",
                "to_unit_id": "u2",
                "metadata": {
                    "source_project": "beta",
                    "sources": [{"url": "https://example.test/a", "source_date": "2026-05-01"}],
                },
            }
        ]
    )

    assert rows(text)[0] == {
        "relation": "supports",
        "relation_type": "Unknown",
        "source_bucket": "beta",
        "edge_count": "1",
        "evidence_count": "1",
        "missing_evidence_count": "0",
        "conflicting_source_count": "0",
        "conflicting_date_count": "0",
        "unit_pairs": "u1->u2",
        "sample_edge_ids": "a",
    }


def test_relation_evidence_consistency_path_mode(tmp_path):
    edges = [edge("1", metadata={"source_id": "doc"})]
    expected = export_relation_evidence_consistency_csv(edges)
    path = tmp_path / "consistency.csv"

    stats = export_relation_evidence_consistency_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["edge_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
