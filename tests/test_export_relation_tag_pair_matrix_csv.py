from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_relation_tag_pair_matrix_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_relation_tag_pair_matrix_csv_expands_tag_combinations_and_skips_missing_units():
    text = export_relation_tag_pair_matrix_csv(
        [
            {"id": "e1", "from_unit_id": "a", "to_unit_id": "b", "relation": "rel", "source": "src", "weight": 2},
            {"id": "e2", "from_unit_id": "missing", "to_unit_id": "b", "relation": "rel", "source": "src"},
        ],
        [
            {"id": "a", "source_id": "sa", "tags": ["x", "y"], "metadata": {}},
            {"id": "b", "source_id": "sb", "tags": ["z"], "metadata": {}},
        ],
    )

    assert rows(text) == [
        {"relation": "rel", "source": "src", "from_tag": "x", "to_tag": "z", "edge_count": "1", "total_weight": "2.00", "sample_edges": "e1"},
        {"relation": "rel", "source": "src", "from_tag": "y", "to_tag": "z", "edge_count": "1", "total_weight": "2.00", "sample_edges": "e1"},
    ]


def test_export_relation_tag_pair_matrix_csv_untagged_and_path_mode(tmp_path):
    path = tmp_path / "pairs.csv"
    stats = export_relation_tag_pair_matrix_csv(
        [{"id": "e1", "from_unit_id": "sa", "to_unit_id": "b"}],
        [{"id": "a", "source_id": "sa", "tags": []}, {"id": "b", "tags": ["tag"]}],
        path,
    )

    assert rows(path.read_text(encoding="utf-8"))[0]["from_tag"] == "Untagged"
    assert stats["edge_count"] == 1
    assert stats["unit_count"] == 2
    assert stats["rows_exported"] == 1


def test_export_relation_tag_pair_matrix_csv_can_exclude_untagged():
    text = export_relation_tag_pair_matrix_csv(
        [{"id": "e1", "from_unit_id": "a", "to_unit_id": "b"}],
        [{"id": "a", "tags": []}, {"id": "b", "tags": ["tag"]}],
        include_untagged=False,
    )

    assert rows(text) == []
