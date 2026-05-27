from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_inline_code_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_single_and_multi_backtick_inline_code():
    result = rows(export_units_to_markdown_inline_code_csv([{"id": "u", "content": "Use `x` and ``y ` z``."}]))
    assert [(row["code"], row["delimiter_length"], row["start_column"]) for row in result] == [("x", "1", "5"), ("y ` z", "2", "13")]


def test_ignores_fenced_code_and_unclosed_delimiters():
    result = rows(export_units_to_markdown_inline_code_csv([{"id": "u", "content": "```py\n`skip`\n```\nkeep `ok`\nbad `open"}]))
    assert [row["code"] for row in result] == ["ok"]
