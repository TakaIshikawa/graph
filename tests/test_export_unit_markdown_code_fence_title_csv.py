from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_code_fence_title_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_code_fence_title_csv_exports_backtick_and_tilde_title_attributes():
    text = export_unit_markdown_code_fence_title_csv(
        [{"id": "u", "title": "T", "source": "s", "content": '```python title="app.py"\nprint(1)\n```\n~~~js name=widget\nx\n~~~'}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "T", "source": "s", "start_line": "1", "end_line": "3", "language": "python", "title_attribute": "app.py", "attribute_name": "title"},
        {"unit_id": "u", "title": "T", "source": "s", "start_line": "4", "end_line": "6", "language": "js", "title_attribute": "widget", "attribute_name": "name"},
    ]


def test_code_fence_title_csv_exports_filename_and_unclosed_end_line():
    text = export_unit_markdown_code_fence_title_csv([{"id": "u", "content": "``` filename=notes.md\nbody"}])

    assert _rows(text) == [{"unit_id": "u", "title": "", "source": "", "start_line": "1", "end_line": "2", "language": "", "title_attribute": "notes.md", "attribute_name": "filename"}]
