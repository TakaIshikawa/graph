from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_inline_code_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_inline_code_ignores_fences_and_handles_multibacktick_empty_shell(tmp_path):
    text = "Use `git status` and ``a ` tick`` and ``.\n```\n`ignored`\n```\n"
    output = tmp_path / "inline.csv"
    result = export_units_to_inline_code_inventory_csv([{"id": "u", "content": text}], output)
    row = rows(output.read_text(encoding="utf-8"))[0]

    assert result["rows_exported"] == 1
    assert row["inline_code_span_count"] == "3"
    assert row["empty_inline_code_count"] == "1"
    assert row["has_shell_like_inline_code"] == "true"
