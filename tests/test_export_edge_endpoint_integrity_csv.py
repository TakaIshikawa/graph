from __future__ import annotations

import csv
from io import StringIO

from graph.export.edge_endpoint_integrity_csv import export_edge_endpoint_integrity_csv
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def unit(unit_id: str, *, source_id: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=SourceProject.MARKDOWN_NOTES,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="note",
        title=unit_id,
        content="",
        tags=[],
        metadata={},
    )


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.RELATES_TO,
    source: EdgeSource | str = EdgeSource.INFERRED,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=source,
        metadata={},
        weight=1.0,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_endpoint_integrity_csv_empty_input_has_header_only():
    assert export_edge_endpoint_integrity_csv([], []) == (
        "edge_id,relation,source,from_unit_id,to_unit_id,missing_from_unit,missing_to_unit,issue_count\n"
    )


def test_edge_endpoint_integrity_csv_omits_edges_with_resolvable_endpoints():
    units = [unit("u1"), unit("u2", source_id="external-2")]
    text = export_edge_endpoint_integrity_csv(
        units,
        [
            edge("valid-by-id", "u1", "u2"),
            edge("valid-by-source", "source-u1", "external-2"),
        ],
    )

    assert rows(text) == []


def test_edge_endpoint_integrity_csv_reports_missing_endpoint_sides():
    units = [unit("u1"), unit("u2")]
    text = export_edge_endpoint_integrity_csv(
        units,
        [
            edge("missing-to", "u1", "absent-to", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE),
            edge("missing-from", "absent-from", "u2", relation=EdgeRelation.BUILDS_ON, source=EdgeSource.MANUAL),
            edge("missing-both", "absent-a", "absent-b", relation=EdgeRelation.REFERENCES, source=EdgeSource.INFERRED),
        ],
    )

    assert rows(text) == [
        {
            "edge_id": "missing-both",
            "relation": "references",
            "source": "inferred",
            "from_unit_id": "absent-a",
            "to_unit_id": "absent-b",
            "missing_from_unit": "true",
            "missing_to_unit": "true",
            "issue_count": "2",
        },
        {
            "edge_id": "missing-from",
            "relation": "builds_on",
            "source": "manual",
            "from_unit_id": "absent-from",
            "to_unit_id": "u2",
            "missing_from_unit": "true",
            "missing_to_unit": "false",
            "issue_count": "1",
        },
        {
            "edge_id": "missing-to",
            "relation": "references",
            "source": "source",
            "from_unit_id": "u1",
            "to_unit_id": "absent-to",
            "missing_from_unit": "false",
            "missing_to_unit": "true",
            "issue_count": "1",
        },
    ]


def test_edge_endpoint_integrity_csv_orders_deterministically_by_severity_relation_edge_and_endpoints():
    units = [unit("u1")]
    edges = [
        edge("b", "x", "a", relation="custom"),
        edge("a", "z", "a", relation="custom"),
        edge("c", "x", "u1", relation="alpha"),
        edge("a", "x", "b", relation="custom"),
    ]

    assert [row["edge_id"] for row in rows(export_edge_endpoint_integrity_csv(units, edges))] == [
        "a",
        "a",
        "b",
        "c",
    ]
    assert export_edge_endpoint_integrity_csv(units, list(reversed(edges))) == export_edge_endpoint_integrity_csv(
        units, edges
    )


def test_edge_endpoint_integrity_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "edge-endpoint-integrity.csv"
    units = [unit("u1")]
    edges = [edge("a", "u1", "missing")]

    expected = export_edge_endpoint_integrity_csv(units, edges)
    stats = export_edge_endpoint_integrity_csv(units, edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "edge_count": 1,
        "problem_edge_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
