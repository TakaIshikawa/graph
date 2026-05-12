from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO

from graph.export import export_edge_coverage_csv
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

TIME = datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc)


def unit(unit_id: str, title: str):
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
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


def test_edge_coverage_computes_degrees_and_relation_summaries():
    text = export_edge_coverage_csv(
        [unit("a", "Alpha"), unit("b", "Beta"), unit("c", "Gamma")],
        [
            edge("ab", "a", "b", EdgeRelation.REFERENCES),
            edge("ac", "a", "c", EdgeRelation.RELATES_TO),
            edge("ba", "b", "a", EdgeRelation.BUILDS_ON),
            edge("xa", "external", "a", EdgeRelation.CHALLENGES),
        ],
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "source": "max",
            "type": "note",
            "in_degree": "2",
            "out_degree": "2",
            "total_degree": "4",
            "relations_in": "builds_on (1); challenges (1)",
            "relations_out": "references (1); relates_to (1)",
            "is_isolate": "false",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "source": "max",
            "type": "note",
            "in_degree": "1",
            "out_degree": "1",
            "total_degree": "2",
            "relations_in": "references (1)",
            "relations_out": "builds_on (1)",
            "is_isolate": "false",
        },
        {
            "unit_id": "c",
            "title": "Gamma",
            "source": "max",
            "type": "note",
            "in_degree": "1",
            "out_degree": "0",
            "total_degree": "1",
            "relations_in": "relates_to (1)",
            "relations_out": "",
            "is_isolate": "false",
        },
    ]


def test_edge_coverage_can_exclude_isolates_and_write_stats(tmp_path):
    path = tmp_path / "coverage.csv"
    stats = export_edge_coverage_csv(
        [unit("a", "Alpha"), unit("b", "Beta")],
        [edge("ab", "a", "b")],
        path,
        include_isolates=False,
    )

    assert stats["rows_exported"] == 2
    assert stats["bytes_written"] == path.stat().st_size
    assert [row["unit_id"] for row in rows(path.read_text(encoding="utf-8"))] == ["a", "b"]


def test_edge_coverage_excludes_isolate_units():
    text = export_edge_coverage_csv(
        [unit("a", "Alpha"), unit("b", "Beta"), unit("c", "Gamma")],
        [edge("ab", "a", "b")],
        include_isolates=False,
    )

    assert [row["unit_id"] for row in rows(text)] == ["a", "b"]
