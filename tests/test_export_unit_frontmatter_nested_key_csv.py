from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_frontmatter_nested_key_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_frontmatter_nested_key_csv_flattens_nested_dictionaries():
    text = export_unit_frontmatter_nested_key_csv([{"id": "u1", "title": "One", "metadata": {"author": {"name": "Ada"}}}])

    assert [(row["key_path"], row["value_type"], row["depth"]) for row in rows(text)] == [
        ("metadata.author", "dict", "2"),
        ("metadata.author.name", "str", "3"),
    ]


def test_unit_frontmatter_nested_key_csv_reports_lists_without_recursing_items():
    text = export_unit_frontmatter_nested_key_csv([{"id": "u1", "title": "One", "frontmatter": {"tags": ["a", "b"]}}])

    assert rows(text) == [{"unit_id": "u1", "title": "One", "key_path": "frontmatter.tags", "value_type": "list", "depth": "2"}]


def test_unit_frontmatter_nested_key_csv_deterministic_depth_and_type():
    text = export_unit_frontmatter_nested_key_csv([{"id": "u1", "title": "One", "metadata": {"z": 1, "a": None}}])

    assert [(row["key_path"], row["value_type"], row["depth"]) for row in rows(text)] == [
        ("metadata.a", "null", "2"),
        ("metadata.z", "int", "2"),
    ]
