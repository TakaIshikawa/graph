from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_edge_directionality_summary_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_edge_directionality_summary_csv_empty_input_returns_header():
    assert export_edge_directionality_summary_csv([]) == "edge_type,total_edges,reciprocal_pairs,one_way_edges,self_loop_edges\n"


def test_export_edge_directionality_summary_csv_counts_directionality_without_double_counting():
    text = export_edge_directionality_summary_csv(
        [
            {"id": "e1", "from_unit_id": "a", "to_unit_id": "b", "relation": "related"},
            {"id": "e2", "from_unit_id": "b", "to_unit_id": "a", "relation": "related"},
            {"id": "e3", "from_unit_id": "a", "to_unit_id": "c", "relation": "related"},
            {"id": "e4", "from_unit_id": "d", "to_unit_id": "d", "relation": "related"},
        ]
    )

    assert rows(text) == [
        {"edge_type": "related", "total_edges": "4", "reciprocal_pairs": "1", "one_way_edges": "1", "self_loop_edges": "1"}
    ]


def test_export_edge_directionality_summary_csv_unknown_type_and_path_mode(tmp_path):
    path = tmp_path / "directionality.csv"
    stats = export_edge_directionality_summary_csv([{"from_unit_id": "a", "to_unit_id": "b"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["edge_type"] == "unknown"
    assert stats["edge_count"] == 1
