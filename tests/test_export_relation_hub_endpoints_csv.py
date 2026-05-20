from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export import export_relation_hub_endpoints_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_hub_endpoints_csv_counts_inbound_outbound_and_neighbors():
    text = export_relation_hub_endpoints_csv(
        [
            {"from_unit_id": "hub", "to_unit_id": "b", "relation": "rel", "source": "src"},
            {"from_unit_id": "hub", "to_unit_id": "c", "relation": "rel", "source": "src"},
            {"from_unit_id": "b", "to_unit_id": "hub", "relation": "rel", "source": "src"},
        ],
        neighbor_limit=1,
    )

    assert rows(text)[0] == {"relation": "rel", "source": "src", "unit_id": "hub", "inbound_count": "1", "outbound_count": "2", "total_count": "3", "distinct_neighbor_count": "2", "top_neighbor_ids": "b"}


def test_export_relation_hub_endpoints_csv_path_mode(tmp_path):
    path = tmp_path / "hubs.csv"
    stats = export_relation_hub_endpoints_csv([{"from_unit_id": "a", "to_unit_id": "b"}], path, min_total_count=1)

    assert len(rows(path.read_text(encoding="utf-8"))) == 2
    assert stats["edge_count"] == 1
    assert stats["rows_exported"] == 2
    assert stats["neighbor_limit"] == 5


def test_export_relation_hub_endpoints_csv_validates_limits():
    with pytest.raises(ValueError, match="neighbor_limit"):
        export_relation_hub_endpoints_csv([], neighbor_limit=0)
