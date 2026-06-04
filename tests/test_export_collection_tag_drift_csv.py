from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_collection_tag_drift_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_collection_tag_drift_uses_collection_unit_ids_and_sorts_by_date():
    text = export_collection_tag_drift_csv(
        [{"id": "c1", "unit_ids": ["u1", "u2", "u3", "u4"]}],
        [
            {"id": "u4", "metadata": {"tags": ["old", "new"], "created_at": "2024-04-01"}},
            {"id": "u1", "metadata": {"tags": ["old"], "created_at": "2024-01-01"}},
            {"id": "u3", "metadata": {"tags": ["new"], "created_at": "2024-03-01"}},
            {"id": "u2", "metadata": {"tags": ["old"], "created_at": "2024-02-01"}},
        ],
    )

    assert rows(text) == [
        {"collection_id": "c1", "tag": "new", "early_count": "0", "late_count": "2", "delta": "2", "drift_status": "emerging", "unit_ids": "u3; u4"},
        {"collection_id": "c1", "tag": "old", "early_count": "2", "late_count": "1", "delta": "-1", "drift_status": "one_sided", "unit_ids": "u1; u2; u4"},
    ]


def test_collection_tag_drift_uses_unit_collection_ids_and_path_mode(tmp_path):
    path = tmp_path / "drift.csv"
    stats = export_collection_tag_drift_csv(
        [{"id": "c"}],
        [
            {"id": "u1", "tags": ["a", "a"], "collection_id": "c"},
            {"id": "u2", "metadata": {"tags": ["b"], "collection_ids": ["c"]}},
        ],
        path,
    )

    assert stats["rows_exported"] == 2
    assert path.read_text(encoding="utf-8").startswith("collection_id,tag,early_count,late_count,delta,drift_status,unit_ids")
