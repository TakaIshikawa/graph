from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_tag_cooccurrence_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_tag_cooccurrence_deduplicates_tags_and_sorts_pairs():
    text = export_units_to_tag_cooccurrence_csv(
        [
            {"id": "b", "metadata": {"tags": ["Beta", "alpha", "beta"]}},
            {"id": "a", "metadata": {"tags": "alpha, gamma"}},
        ]
    )

    assert rows(text) == [
        {"tag_a": "alpha", "tag_b": "beta", "unit_count": "1", "unit_ids": "b"},
        {"tag_a": "alpha", "tag_b": "gamma", "unit_count": "1", "unit_ids": "a"},
    ]


def test_tag_cooccurrence_path_mode(tmp_path):
    path = tmp_path / "tags.csv"
    stats = export_units_to_tag_cooccurrence_csv([{"id": "a", "metadata": {"tags": ["x", "y"]}}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["unit_ids"] == "a"
    assert stats["rows_exported"] == 1
