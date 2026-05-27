from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_frontmatter_duplicate_key_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_repeated_scalar_keys_preserve_duplicate_spelling():
    content = "---\ntitle: One\nTitle: Two\ntitle: Three\n---\nbody"

    found = rows(export_unit_frontmatter_duplicate_key_csv([{"id": "u", "content": content}]))

    assert [(row["key"], row["first_line_number"], row["duplicate_line_number"], row["occurrence_count"]) for row in found] == [("Title", "2", "3", "2"), ("title", "2", "4", "3")]


def test_nested_looking_keys_and_non_frontmatter_content():
    content = "---\nparent:\n  child: one\n  child: two\n---\n---\nchild: outside\nchild: outside again"

    found = rows(export_unit_frontmatter_duplicate_key_csv([{"id": "u", "content": content}]))

    assert [(row["key"], row["duplicate_line_number"]) for row in found] == [("child", "4")]


def test_sorting_and_file_writing(tmp_path):
    units = [
        {"id": "b", "content": "---\nx: 1\nx: 2\n---"},
        {"id": "a", "content": "---\ny: 1\ny: 2\n---"},
    ]
    path = tmp_path / "duplicates.csv"

    result = export_unit_frontmatter_duplicate_key_csv(units, path)
    found = rows(path.read_text())

    assert [row["unit_id"] for row in found] == ["a", "b"]
    assert result["rows_exported"] == 2
    assert result["bytes_written"] == path.stat().st_size
