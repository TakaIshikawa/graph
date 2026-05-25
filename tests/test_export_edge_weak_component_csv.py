from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_edge_weak_component_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_weak_component_csv_computes_stable_weak_components():
    text = export_edge_weak_component_csv(
        [
            {"source_id": "b", "target_id": "c"},
            {"source_id": "a", "target_id": "b"},
            {"source_id": "x", "target_id": "y"},
        ]
    )

    assert rows(text) == [
        {"component_id": "a", "node_count": "3", "edge_count": "2", "source_nodes": "a", "sink_nodes": "c", "representative_nodes": "a; b; c", "density": "0.3333"},
        {"component_id": "x", "node_count": "2", "edge_count": "1", "source_nodes": "x", "sink_nodes": "y", "representative_nodes": "x; y", "density": "0.5000"},
    ]


def test_edge_weak_component_csv_includes_optional_isolated_nodes():
    text = export_edge_weak_component_csv([], nodes=[{"id": "solo"}])

    assert rows(text) == [
        {"component_id": "solo", "node_count": "1", "edge_count": "0", "source_nodes": "", "sink_nodes": "", "representative_nodes": "solo", "density": "0.0000"}
    ]
