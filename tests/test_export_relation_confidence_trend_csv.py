from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_relation_confidence_trend_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_relation_confidence_trend_groups_by_month_and_relation_type():
    text = export_relation_confidence_trend_csv(
        [
            {"relation_type": "supports", "metadata": {"confidence": 0.4, "timestamp": "2025-01-10"}},
            {"relation_type": "supports", "metadata": {"confidence": 0.8, "timestamp": "2025-01-20"}},
            SimpleNamespace(relation="contradicts", weight=0.9, metadata={"created_at": "2025-02-01"}),
        ]
    )

    by_key = {(row["period"], row["relation_type"]): row for row in rows(text)}
    assert by_key[("2025-01", "supports")]["relation_count"] == "2"
    assert by_key[("2025-01", "supports")]["average_confidence"] == "0.60"
    assert by_key[("2025-01", "supports")]["low_confidence_count"] == "1"
    assert by_key[("2025-02", "contradicts")]["max_confidence"] == "0.90"


def test_relation_confidence_trend_writes_path_metadata(tmp_path):
    path = tmp_path / "trend.csv"
    stats = export_relation_confidence_trend_csv([{"relation": "related", "confidence": 0.7}], path)

    assert stats["relation_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
