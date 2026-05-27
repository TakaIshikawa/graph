from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_frontmatter_alias_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_frontmatter_alias_csv_expands_list_aliases_with_indexes():
    text = export_unit_frontmatter_alias_csv([{"id": "u1", "title": "One", "metadata": {"aliases": ["Alpha", "Beta"]}}])

    assert [(row["alias"], row["index"], row["source_field"]) for row in rows(text)] == [
        ("Alpha", "0", "metadata.aliases"),
        ("Beta", "1", "metadata.aliases"),
    ]


def test_unit_frontmatter_alias_csv_exports_scalar_alias():
    text = export_unit_frontmatter_alias_csv([{"id": "u1", "title": "One", "frontmatter": {"alias": "Solo"}}])

    assert rows(text)[0]["alias"] == "Solo"
    assert rows(text)[0]["source_field"] == "frontmatter.alias"


def test_unit_frontmatter_alias_csv_missing_aliases_header_only():
    assert export_unit_frontmatter_alias_csv([{"id": "u1", "title": "One", "metadata": {"title": "Nope"}}]) == "unit_id,title,alias,index,source_field\n"
