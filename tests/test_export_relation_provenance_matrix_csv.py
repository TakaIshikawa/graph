from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_provenance_matrix_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.RELATES_TO,
    from_unit_id: str | None = None,
    to_unit_id: str | None = None,
    weight: object = 1.0,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id or f"from-{edge_id}",
        to_unit_id=to_unit_id or f"to-{edge_id}",
        relation=relation,
        source=EdgeSource.INFERRED,
        weight=weight,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_provenance_matrix_csv_empty_input_returns_header():
    assert export_relation_provenance_matrix_csv([]) == (
        "relation,source_project,source_entity_type,edge_count,unique_from_units,unique_to_units,"
        "average_weight,average_confidence\n"
    )


def test_relation_provenance_matrix_csv_groups_by_relation_and_source_metadata():
    text = export_relation_provenance_matrix_csv(
        [
            edge(
                "a",
                relation=EdgeRelation.REFERENCES,
                from_unit_id="u1",
                to_unit_id="u2",
                weight=1,
                metadata={"source_project": "zotero", "source_entity_type": "paper", "confidence": 0.5},
            ),
            edge(
                "b",
                relation=EdgeRelation.REFERENCES,
                from_unit_id="u1",
                to_unit_id="u3",
                weight=3,
                metadata={"source_project": "zotero", "source_entity_type": "paper", "confidence": 1.0},
            ),
        ]
    )

    assert rows(text) == [
        {
            "relation": "references",
            "source_project": "zotero",
            "source_entity_type": "paper",
            "edge_count": "2",
            "unique_from_units": "1",
            "unique_to_units": "2",
            "average_weight": "2.00",
            "average_confidence": "0.75",
        }
    ]


def test_relation_provenance_matrix_csv_uses_unknown_and_ignores_non_numeric_averages():
    text = export_relation_provenance_matrix_csv(
        [
            edge("a", weight="heavy", metadata={"confidence": "high"}),
            edge("b", weight=2, metadata={"confidence": 0.25}),
        ]
    )

    assert rows(text)[0] == {
        "relation": "relates_to",
        "source_project": "Unknown",
        "source_entity_type": "Unknown",
        "edge_count": "2",
        "unique_from_units": "2",
        "unique_to_units": "2",
        "average_weight": "2.00",
        "average_confidence": "0.25",
    }


def test_relation_provenance_matrix_csv_sorts_deterministically_and_path_mode(tmp_path):
    path = tmp_path / "provenance.csv"
    edges = [
        edge("b", relation=EdgeRelation.RELATES_TO, metadata={"source_project": "b", "source_entity_type": "note"}),
        edge("a", relation=EdgeRelation.BUILDS_ON, metadata={"source_project": "a", "source_entity_type": "note"}),
    ]

    assert export_relation_provenance_matrix_csv(edges) == export_relation_provenance_matrix_csv(reversed(edges))
    expected = export_relation_provenance_matrix_csv(edges)
    stats = export_relation_provenance_matrix_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 2,
        "rows_exported": 2,
        "bytes_written": path.stat().st_size,
    }
