from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_code_fence_filename_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_code_fence_filename_csv_detects_supported_attributes():
    text = export_unit_code_fence_filename_csv(
        [{"id": "u1", "title": "One", "content": '```python filename=app.py\n```\n```js file="main.js"\n```\n``` title="README.md"'}]
    )

    assert [(row["language"], row["filename"], row["attribute_name"], row["line"]) for row in rows(text)] == [
        ("python", "app.py", "filename", "1"),
        ("js", "main.js", "file", "3"),
        ("", "README.md", "title", "5"),
    ]


def test_unit_code_fence_filename_csv_ignores_language_only_fences():
    assert rows(export_unit_code_fence_filename_csv([{"id": "u1", "title": "One", "content": "```python\nx\n```"}])) == []


def test_unit_code_fence_filename_csv_supports_quoted_and_unquoted_values():
    text = export_unit_code_fence_filename_csv([{"id": "u1", "title": "One", "content": '``` filename="src/app.py"\n``` file=test.py'}])

    assert [row["filename"] for row in rows(text)] == ["src/app.py", "test.py"]
