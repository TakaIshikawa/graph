from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_relation_direction_imbalance_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_direction_imbalance_csv_counts_direction_coverage():
    text = export_relation_direction_imbalance_csv(
        [
            {"from_unit_id": "a", "to_unit_id": "b", "relation": "rel", "source": "src"},
            {"from_unit_id": "b", "to_unit_id": "a", "relation": "rel", "source": "src"},
            {"from_unit_id": "a", "to_unit_id": "c", "relation": "rel", "source": "src"},
            {"from_unit_id": "d", "to_unit_id": "a", "relation": "rel", "source": "src"},
        ]
    )

    assert rows(text) == [
        {"relation": "rel", "source": "src", "pair_count": "3", "forward_only_count": "1", "reverse_only_count": "1", "bidirectional_count": "1", "imbalance_percent": "0.00", "sample_pairs": "a->b; a->c; a->d"}
    ]


def test_export_relation_direction_imbalance_csv_filters_and_path_mode(tmp_path):
    path = tmp_path / "imbalance.csv"
    stats = export_relation_direction_imbalance_csv([{"from_unit_id": "a", "to_unit_id": "b"}], path, min_pair_count=2)

    assert rows(path.read_text(encoding="utf-8")) == []
    assert stats["edge_count"] == 1
    assert stats["rows_exported"] == 0
    assert stats["min_pair_count"] == 2


def test_export_relation_direction_imbalance_csv_validates_min_pair_count():
    with pytest.raises(ValueError, match="min_pair_count"):
        export_relation_direction_imbalance_csv([], min_pair_count=0)
