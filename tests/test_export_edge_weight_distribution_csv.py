from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_edge_weight_distribution_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_weight_distribution_uses_documented_buckets():
    text = export_edge_weight_distribution_csv(
        [
            {"relation": "rel", "weight": 0.1},
            {"relation": "rel", "weight": 0.34},
            {"relation": "rel", "weight": 0.67},
            {"relation": "rel"},
            {"type": "other", "weight": "bad"},
        ]
    )

    assert rows(text) == [
        {
            "edge_type": "other",
            "edge_count": "1",
            "min_weight": "",
            "max_weight": "",
            "average_weight": "",
            "low_count": "0",
            "medium_count": "0",
            "high_count": "0",
            "missing_weight_count": "1",
        },
        {
            "edge_type": "rel",
            "edge_count": "4",
            "min_weight": "0.10",
            "max_weight": "0.67",
            "average_weight": "0.37",
            "low_count": "1",
            "medium_count": "1",
            "high_count": "1",
            "missing_weight_count": "1",
        },
    ]


def test_edge_weight_distribution_path_mode(tmp_path):
    path = tmp_path / "weights.csv"
    stats = export_edge_weight_distribution_csv([{"relation": "rel", "weight": 1}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["high_count"] == "1"
    assert stats["edge_count"] == 1
