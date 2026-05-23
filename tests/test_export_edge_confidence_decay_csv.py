from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_edge_confidence_decay_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_edge_confidence_decay_csv_uses_reference_date():
    text = export_edge_confidence_decay_csv(
        [
            {"id": "e2", "from_unit_id": "b", "to_unit_id": "c", "metadata": {"confidence": "bad"}},
            {"id": "e1", "from_unit_id": "a", "to_unit_id": "b", "confidence": 0.5, "metadata": {"updated_at": "2023-01-01T00:00:00Z"}},
        ],
        reference_date="2024-01-01T00:00:00Z",
    )

    assert rows(text) == [
        {
            "edge_id": "e1",
            "from_unit_id": "a",
            "to_unit_id": "b",
            "confidence": "0.50",
            "age_days": "365",
            "confidence_decay_score": "0.25",
        },
        {
            "edge_id": "e2",
            "from_unit_id": "b",
            "to_unit_id": "c",
            "confidence": "1.00",
            "age_days": "0",
            "confidence_decay_score": "1.00",
        },
    ]


def test_export_edge_confidence_decay_csv_path_mode(tmp_path):
    path = tmp_path / "decay.csv"
    stats = export_edge_confidence_decay_csv([{"id": "e1", "weight": 0.25}], path, reference_date="2024-01-01T00:00:00Z")

    assert rows(path.read_text(encoding="utf-8"))[0]["confidence"] == "0.25"
    assert stats["edge_count"] == 1
    assert stats["rows_exported"] == 1
