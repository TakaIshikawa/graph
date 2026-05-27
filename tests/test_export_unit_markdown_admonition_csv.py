from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_admonition_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_fenced_admonition_syntax():
    [row] = rows(export_units_to_markdown_admonition_csv([{"id": "u", "content": "!!! note Remember\n    Body"}]))
    assert row["syntax"] == "admonition"
    assert row["kind"] == "note"
    assert row["first_text"] == "Remember"


def test_obsidian_callout_syntax():
    [row] = rows(export_units_to_markdown_admonition_csv([{"id": "u", "content": "> [!TIP] Use this\n> text"}]))
    assert row["syntax"] == "obsidian_callout"
    assert row["kind"] == "tip"
    assert row["marker"] == "> [!TIP] Use this"


def test_empty_input_has_header():
    assert export_units_to_markdown_admonition_csv([]) == "unit_id,title,syntax,kind,marker,start_line,first_text\n"
