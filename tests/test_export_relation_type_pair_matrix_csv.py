from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_type_pair_matrix_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_type_pair_matrix_csv_empty_input_returns_header():
    assert export_relation_type_pair_matrix_csv([], []) == "relation_type,source_type,target_type,count,relation_ids\n"


def test_export_relation_type_pair_matrix_csv_groups_relation_and_endpoint_types():
    text = export_relation_type_pair_matrix_csv(
        [{"id": "a", "source_entity_type": "note"}, {"id": "b", "source_entity_type": "task"}],
        [
            {"id": "r1", "from_unit_id": "a", "to_unit_id": "b", "relation": "references"},
            {"id": "r2", "from_unit_id": "a", "to_unit_id": "b", "relation": "references"},
            {"id": "r3", "source_type": "task", "target_type": "note", "relation_type": "blocks"},
        ],
    )

    assert rows(text) == [
        {"relation_type": "blocks", "source_type": "task", "target_type": "note", "count": "1", "relation_ids": "r3"},
        {"relation_type": "references", "source_type": "note", "target_type": "task", "count": "2", "relation_ids": "r1; r2"},
    ]


def test_export_relation_type_pair_matrix_csv_uses_unknown_buckets_and_path_mode(tmp_path):
    path = tmp_path / "relation-matrix.csv"
    stats = export_relation_type_pair_matrix_csv([], [{"id": "r1"}], path)

    assert rows(path.read_text(encoding="utf-8")) == [
        {"relation_type": "unknown", "source_type": "unknown", "target_type": "unknown", "count": "1", "relation_ids": "r1"}
    ]
    assert stats["relation_count"] == 1
