from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_edge_type_matrix_csv
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

TIME = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


def unit(unit_id: str, kind: str):
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type=kind,
        title=unit_id,
        content="content",
        content_type=ContentType.INSIGHT,
        created_at=TIME,
        ingested_at=TIME,
        updated_at=TIME,
    )


def edge(edge_id: str, source: str, target: str, relation=EdgeRelation.RELATES_TO):
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=source,
        to_unit_id=target,
        relation=relation,
        source=EdgeSource.INFERRED,
        created_at=TIME,
    )


def rows(text: str):
    return list(csv.DictReader(StringIO(text)))


def test_edge_type_matrix_aggregates_relation_source_target_type_counts():
    text = export_edge_type_matrix_csv(
        [unit("a", "note"), unit("b", "quote"), unit("c", "quote")],
        [
            edge("ab", "a", "b", EdgeRelation.REFERENCES),
            edge("ac", "a", "c", EdgeRelation.REFERENCES),
            edge("ba", "b", "a", EdgeRelation.BUILDS_ON),
        ],
    )

    assert rows(text) == [
        {"relation": "builds_on", "source_type": "quote", "target_type": "note", "count": "1"},
        {"relation": "references", "source_type": "note", "target_type": "quote", "count": "2"},
    ]


def test_edge_type_matrix_skips_missing_endpoints_and_reports_stats(tmp_path):
    path = tmp_path / "matrix.csv"
    stats = export_edge_type_matrix_csv(
        [unit("a", "note"), unit("b", "quote")],
        [edge("ab", "a", "b"), edge("missing", "a", "missing")],
        path,
    )

    assert stats["skipped_edges"] == 1
    assert stats["rows_exported"] == 1
    assert rows(path.read_text(encoding="utf-8"))[0]["count"] == "1"


def test_edge_type_matrix_is_deterministic_and_can_include_zeroes():
    units = [unit("b", "quote"), unit("a", "note")]
    edges = [edge("ab", "a", "b", EdgeRelation.REFERENCES)]

    first = export_edge_type_matrix_csv(units, edges, include_zeroes=True)
    second = export_edge_type_matrix_csv(reversed(units), reversed(edges), include_zeroes=True)

    assert first == second
    assert {"relation": "references", "source_type": "quote", "target_type": "note", "count": "0"} in rows(first)
