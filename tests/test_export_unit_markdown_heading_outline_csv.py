from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_heading_outline_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_heading_outline_ignores_code_fences_and_preserves_order():
    result = rows(export_units_to_markdown_heading_outline_csv([
        {"id": "u", "content": "# A\n## B\n```\n# No\n```\n# C\n### D"},
        {"id": "z", "content": "none"},
    ]))

    first = {row["unit_id"]: row for row in result}["u"]
    assert first["heading_count"] == "4"
    assert first["max_depth"] == "3"
    assert first["top_level_headings"] == "A; C"
    assert first["deepest_heading"] == "D"
    assert {row["unit_id"]: row for row in result}["z"]["has_outline"] == "false"
