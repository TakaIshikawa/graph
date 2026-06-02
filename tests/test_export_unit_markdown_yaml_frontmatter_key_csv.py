from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_yaml_frontmatter_keys_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_yaml_frontmatter_key_csv_exports_top_level_keys_with_value_state():
    text = export_unit_markdown_yaml_frontmatter_keys_to_csv(
        [
            {"id": "u", "title": "Unit", "content": "---\ntitle: A\ntags:\n  - x\nempty:\n---\nbody\n---\nignored: yes\n---"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "u", "title": "Unit", "line_number": "2", "key": "title", "has_value": "true"},
        {"unit_id": "u", "title": "Unit", "line_number": "3", "key": "tags", "has_value": "true"},
        {"unit_id": "u", "title": "Unit", "line_number": "5", "key": "empty", "has_value": "false"},
    ]


def test_yaml_frontmatter_key_csv_ignores_non_leading_thematic_breaks(tmp_path):
    path = tmp_path / "frontmatter.csv"

    stats = export_unit_markdown_yaml_frontmatter_keys_to_csv([{"id": "u", "content": "body\n---\ntitle: no\n---"}], path)

    assert rows(path.read_text(encoding="utf-8")) == []
    assert stats["rows_exported"] == 0
