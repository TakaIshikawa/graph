from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_edge_weight_outliers_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_edge_weight_outliers_csv_flags_high_and_low_weights():
    text = export_edge_weight_outliers_csv(
        [
            {"id": "low", "from_unit_id": "a", "to_unit_id": "b", "relation": "rel", "source": "src", "weight": -10},
            {"id": "mid1", "from_unit_id": "b", "to_unit_id": "c", "relation": "rel", "source": "src", "weight": 1},
            {"id": "mid2", "from_unit_id": "c", "to_unit_id": "d", "relation": "rel", "source": "src", "weight": 1},
            {"id": "high", "from_unit_id": "d", "to_unit_id": "e", "relation": "rel", "source": "src", "weight": 10},
        ],
        zscore_threshold=1.0,
    )

    data = rows(text)
    assert [row["edge_id"] for row in data] == ["high", "low"]
    assert data[0]["direction"] == "high"
    assert data[1]["direction"] == "low"
    assert data[0]["weight"] == "10.00"


def test_export_edge_weight_outliers_csv_skips_small_or_zero_variance_groups_and_path_mode(tmp_path):
    path = tmp_path / "outliers.csv"
    stats = export_edge_weight_outliers_csv(
        [{"id": "a", "weight": 1, "from_unit_id": "a", "to_unit_id": "b"}, {"id": "b", "weight": 1, "from_unit_id": "b", "to_unit_id": "c"}],
        path,
    )

    assert rows(path.read_text(encoding="utf-8")) == []
    assert stats["edge_count"] == 2
    assert stats["rows_exported"] == 0
    assert stats["min_group_size"] == 3


def test_export_edge_weight_outliers_csv_validates_limits():
    with pytest.raises(ValueError, match="zscore_threshold"):
        export_edge_weight_outliers_csv([], zscore_threshold=0)
