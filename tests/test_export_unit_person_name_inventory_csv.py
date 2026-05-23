from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_person_name_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_person_name_inventory_csv_expands_lists_and_delimited_strings():
    text = export_unit_person_name_inventory_csv(
        [
            {"id": "u2", "metadata": {"authors": ["Ada Lovelace", "Grace Hopper"]}},
            {"id": "u1", "metadata": {"participants": "Alan Turing; Ada Lovelace"}},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u1", "person_name": "Ada Lovelace", "source_field": "participants", "normalized_name": "ada lovelace"},
        {"unit_id": "u1", "person_name": "Alan Turing", "source_field": "participants", "normalized_name": "alan turing"},
        {"unit_id": "u2", "person_name": "Ada Lovelace", "source_field": "authors", "normalized_name": "ada lovelace"},
        {"unit_id": "u2", "person_name": "Grace Hopper", "source_field": "authors", "normalized_name": "grace hopper"},
    ]


def test_export_unit_person_name_inventory_csv_path_mode(tmp_path):
    path = tmp_path / "people.csv"
    stats = export_unit_person_name_inventory_csv([{"id": "u1", "author": "Ada"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["normalized_name"] == "ada"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
