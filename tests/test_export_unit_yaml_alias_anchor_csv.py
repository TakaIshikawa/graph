from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_yaml_alias_anchor_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_yaml_alias_anchor_csv_scans_only_leading_frontmatter():
    text = export_units_to_yaml_alias_anchor_csv(
        [{"id": "u", "title": "Doc", "content": "---\ndefaults: &base\ncopy: *base\n---\nbody: &ignored"}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Doc", "line_number": "2", "symbol_type": "anchor", "symbol_name": "base", "key_path": "defaults", "raw_value": "defaults: &base"},
        {"unit_id": "u", "title": "Doc", "line_number": "3", "symbol_type": "alias", "symbol_name": "base", "key_path": "copy", "raw_value": "copy: *base"},
    ]


def test_yaml_alias_anchor_csv_path_mode_counts_units_without_frontmatter(tmp_path):
    path = tmp_path / "yaml.csv"
    units = [{"id": "u", "content": "body: *alias"}]

    expected = export_units_to_yaml_alias_anchor_csv(units)
    stats = export_units_to_yaml_alias_anchor_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 0
    assert stats["bytes_written"] == path.stat().st_size
