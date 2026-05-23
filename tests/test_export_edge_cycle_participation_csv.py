from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_edge_cycle_participation_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_cycle_participation_empty_input_has_header():
    assert (
        export_edge_cycle_participation_csv([])
        == "source_id,target_id,relation,cycle_count,shortest_cycle_length\n"
    )


def test_edge_cycle_participation_counts_simple_cycles_and_zero_edges():
    text = export_edge_cycle_participation_csv(
        [
            {"from_unit_id": "a", "to_unit_id": "b", "relation": "rel"},
            {"from_unit_id": "b", "to_unit_id": "c", "relation": "rel"},
            {"from_unit_id": "c", "to_unit_id": "a", "relation": "rel"},
            {"from_unit_id": "c", "to_unit_id": "d", "relation": "rel"},
        ]
    )

    result = {(row["source_id"], row["target_id"]): row for row in rows(text)}
    assert result[("a", "b")]["cycle_count"] == "1"
    assert result[("a", "b")]["shortest_cycle_length"] == "3"
    assert result[("c", "d")]["cycle_count"] == "0"
    assert result[("c", "d")]["shortest_cycle_length"] == ""


def test_edge_cycle_participation_can_omit_zero_edges(tmp_path):
    path = tmp_path / "cycles.csv"
    stats = export_edge_cycle_participation_csv(
        [{"source_id": "a", "target_id": "b", "relation": "x"}],
        path,
        include_zero=False,
    )

    assert rows(path.read_text(encoding="utf-8")) == []
    assert stats["rows_exported"] == 0
