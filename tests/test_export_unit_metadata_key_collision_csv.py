from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_metadata_key_collision_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_metadata_key_collision_normalizes_case_separators_and_whitespace():
    text = export_unit_metadata_key_collision_csv(
        [{"id": "a", "title": "Alpha", "metadata": {"Source URL": "a", "source_url": "b", "source-url": "", "other": "x"}}]
    )

    assert rows(text) == [
        {"unit_id": "a", "title": "Alpha", "normalized_key": "sourceurl", "original_keys": "Source URL; source-url; source_url", "value_count": "2"}
    ]


def test_unit_metadata_key_collision_path_mode_and_no_collisions(tmp_path):
    path = tmp_path / "collisions.csv"
    stats = export_unit_metadata_key_collision_csv([{"id": "a", "title": "Alpha", "metadata": {"one": 1}}], path)

    assert rows(path.read_text(encoding="utf-8")) == []
    assert stats["rows_exported"] == 0
