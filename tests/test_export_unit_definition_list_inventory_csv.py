from __future__ import annotations

import csv
from io import StringIO
from types import SimpleNamespace

from graph.export import export_units_to_definition_list_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_definition_list_inventory_groups_terms_and_ignores_fences(tmp_path):
    text = "Term\n: one\n: two\nOther\n: single\n```\nFake\n: ignored\n```"
    output = tmp_path / "definitions.csv"
    result = export_units_to_definition_list_inventory_csv([SimpleNamespace(id="u", content=text)], output)
    row = rows(output.read_text(encoding="utf-8"))[0]

    assert result["rows_exported"] == 1
    assert row["definition_term_count"] == "2"
    assert row["definition_line_count"] == "3"
    assert row["multi_definition_term_count"] == "1"
    assert row["max_definition_lines_per_term"] == "2"
