from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO

from graph.export.unit_relation_degree_csv import export_unit_relation_degree_csv
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@dataclass
class UnitLike:
    id: str
    title: str
    source_project: str


@dataclass
class EdgeLike:
    source: str
    target: str


def unit(unit_id: str, title: str | None = None, source_project: str = "Project A") -> KnowledgeUnit:
    return KnowledgeUnit.model_construct(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title or f"Title {unit_id}",
        content="content",
        metadata={},
        tags=[],
    )


def edge(source: str, target: str, edge_id: str = "") -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(id=edge_id, from_unit_id=source, to_unit_id=target, relation="related")


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_relation_degree_csv_empty_input_has_header_only():
    assert export_unit_relation_degree_csv([], []) == (
        "unit_id,title,source_project,in_degree,out_degree,total_degree,"
        "distinct_neighbor_count,neighbor_ids\n"
    )


def test_unit_relation_degree_csv_counts_isolated_bidirectional_and_multiple_edges():
    text = export_unit_relation_degree_csv(
        [
            unit("c", "Gamma", "Project C"),
            unit("a", "Alpha"),
            unit("b", "Beta", "Project B"),
            unit("d", "Delta"),
        ],
        [
            edge("a", "b", "1"),
            edge("b", "a", "2"),
            edge("a", "b", "3"),
            edge("x", "a", "4"),
            edge("b", "z", "5"),
        ],
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "source_project": "Project A",
            "in_degree": "2",
            "out_degree": "2",
            "total_degree": "4",
            "distinct_neighbor_count": "2",
            "neighbor_ids": "b; x",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "source_project": "Project B",
            "in_degree": "2",
            "out_degree": "2",
            "total_degree": "4",
            "distinct_neighbor_count": "2",
            "neighbor_ids": "a; z",
        },
        {
            "unit_id": "c",
            "title": "Gamma",
            "source_project": "Project C",
            "in_degree": "0",
            "out_degree": "0",
            "total_degree": "0",
            "distinct_neighbor_count": "0",
            "neighbor_ids": "",
        },
        {
            "unit_id": "d",
            "title": "Delta",
            "source_project": "Project A",
            "in_degree": "0",
            "out_degree": "0",
            "total_degree": "0",
            "distinct_neighbor_count": "0",
            "neighbor_ids": "",
        },
    ]


def test_unit_relation_degree_csv_accepts_mappings_and_model_like_inputs():
    text = export_unit_relation_degree_csv(
        [
            {"id": "b", "title": "Beta", "source_project": "Project B"},
            UnitLike("a", "Alpha", "Project A"),
            {"source_id": "fallback", "title": "Fallback"},
        ],
        [
            {"source_unit_id": "a", "target_unit_id": "b"},
            EdgeLike("b", "a"),
            {"from": {"id": "fallback"}, "to": {"id": "a"}},
        ],
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "source_project": "Project A",
            "in_degree": "2",
            "out_degree": "1",
            "total_degree": "3",
            "distinct_neighbor_count": "2",
            "neighbor_ids": "b; fallback",
        },
        {
            "unit_id": "b",
            "title": "Beta",
            "source_project": "Project B",
            "in_degree": "1",
            "out_degree": "1",
            "total_degree": "2",
            "distinct_neighbor_count": "1",
            "neighbor_ids": "a",
        },
        {
            "unit_id": "fallback",
            "title": "Fallback",
            "source_project": "Unknown",
            "in_degree": "0",
            "out_degree": "1",
            "total_degree": "1",
            "distinct_neighbor_count": "1",
            "neighbor_ids": "a",
        },
    ]


def test_unit_relation_degree_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "unit-relation-degree.csv"
    units = [unit("a", "Alpha"), unit("b", "Beta")]
    edges = [edge("a", "b")]

    expected = export_unit_relation_degree_csv(units, edges)
    stats = export_unit_relation_degree_csv(units, edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 2,
        "edge_count": 1,
        "rows_exported": 2,
        "bytes_written": path.stat().st_size,
    }
