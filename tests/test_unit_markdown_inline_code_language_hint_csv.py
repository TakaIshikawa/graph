from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_inline_code_language_hint_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_inline_code_language_hints_outside_fences():
    result = rows(export_units_to_markdown_inline_code_language_hint_csv([{"id": "u", "content": "Use `Python: print(1)`.\n```md\n`js: skip()`\n```"}]))

    assert result == [{"unit_id": "u", "title": "", "language": "python", "code": "print(1)", "line_number": "1", "start_column": "5", "excerpt": "Use `Python: print(1)`."}]
