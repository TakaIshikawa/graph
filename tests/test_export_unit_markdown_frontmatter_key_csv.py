from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_frontmatter_key_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_leading_frontmatter_scalar_keys_with_paths_and_lines():
    result = rows(
        export_units_to_markdown_frontmatter_key_csv(
            [{"id": "u", "title": "Title", "source_project": "docs", "content": "---\ntitle: Doc\nrating: 4\ndraft: false\nnested:\n  child: yes\n---\n# Body"}]
        )
    )

    assert [(row["key_path"], row["value_type"], row["line_number"]) for row in result] == [
        ("title", "string", "2"),
        ("rating", "number", "3"),
        ("draft", "boolean", "4"),
        ("nested.child", "string", "6"),
    ]
    assert result[0]["source"] == "docs"


def test_ignores_non_leading_and_unclosed_frontmatter():
    assert rows(export_units_to_markdown_frontmatter_key_csv([{"id": "a", "content": "# Body\n---\ntitle: Later\n---"}])) == []
    assert rows(export_units_to_markdown_frontmatter_key_csv([{"id": "b", "content": "---\ntitle: Open"}])) == []
