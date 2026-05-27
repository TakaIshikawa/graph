from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_temporal_order_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_temporal_order_csv_classifies_order_and_lag_days():
    rows = _rows(export_relation_temporal_order_csv([
        {"id": "r1", "source_id": "a", "target_id": "b", "source_timestamp": "2024-01-01T00:00:00Z", "target_timestamp": "2024-01-03T12:00:00+00:00"},
        {"id": "r2", "source_id": "b", "target_id": "a", "metadata": {"source_timestamp": "2024-01-03", "target_timestamp": "2024-01-01"}},
        {"id": "r3", "source_id": "x", "target_id": "y", "source_timestamp": "bad"},
    ]))

    assert [(row["relation_id"], row["order"], row["lag_days"]) for row in rows] == [
        ("r1", "source_before_target", "2.5"),
        ("r2", "target_before_source", "-2"),
        ("r3", "unknown", ""),
    ]
