from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_yaml_frontmatter_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_simple_nested_and_list_frontmatter_keys():
    content = "---\ntitle: Example\ntags:\n  - One\nmeta:\n  author: Ada\n---\n# Body"
    result = rows(export_units_to_markdown_yaml_frontmatter_csv([{"id": "u", "content": content}]))

    assert [(row["key_path"], row["value_excerpt"], row["value_kind"], row["line_number"]) for row in result] == [
        ("title", "Example", "scalar", "2"),
        ("tags", "", "mapping_or_list", "3"),
        ("tags[]", "One", "list_item", "4"),
        ("meta", "", "mapping_or_list", "5"),
        ("meta.author", "Ada", "scalar", "6"),
    ]


def test_ignores_missing_or_unterminated_frontmatter():
    assert rows(export_units_to_markdown_yaml_frontmatter_csv([{"id": "u", "content": "# Body"}, {"id": "v", "content": "---\ntitle: Nope"}])) == []
