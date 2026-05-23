from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_polarity_conflict_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_polarity_conflict_csv_groups_endpoint_conflicts():
    text = export_relation_polarity_conflict_csv(
        [
            {"id": "r2", "from_unit_id": "b", "to_unit_id": "a", "relation": "claim", "metadata": {"sentiment": "negative"}},
            {"id": "r1", "from_unit_id": "a", "to_unit_id": "b", "relation": "claim", "polarity": "positive"},
            {"id": "r3", "from_unit_id": "a", "to_unit_id": "c", "relation": "claim", "polarity": "positive"},
        ]
    )

    assert rows(text) == [
        {
            "endpoint_pair": "a <-> b",
            "relation_type": "claim",
            "conflicting_polarities": "negative; positive",
            "relation_ids": "r1; r2",
            "conflict_count": "2",
        }
    ]


def test_export_relation_polarity_conflict_csv_path_mode(tmp_path):
    path = tmp_path / "polarity.csv"
    stats = export_relation_polarity_conflict_csv([{"id": "r1", "relation": "supports"}], path)

    assert path.read_text(encoding="utf-8") == "endpoint_pair,relation_type,conflicting_polarities,relation_ids,conflict_count\n"
    assert stats["relation_count"] == 1
    assert stats["rows_exported"] == 0
