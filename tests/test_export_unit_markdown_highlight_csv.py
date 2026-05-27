from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_highlight_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_markdown_highlight_csv_exports_multiple_highlights_on_one_line():
    text = export_unit_markdown_highlight_csv([{"id": "u1", "title": "One", "content": "Use ==alpha== and ==beta==."}])

    assert [row["highlight_text"] for row in rows(text)] == ["alpha", "beta"]


def test_unit_markdown_highlight_csv_ignores_empty_markers():
    text = export_unit_markdown_highlight_csv([{"id": "u1", "title": "One", "content": "Empty ==== but ==kept=="}])

    assert [row["highlight_text"] for row in rows(text)] == ["kept"]


def test_unit_markdown_highlight_csv_exports_context_deterministically():
    text = export_unit_markdown_highlight_csv(
        [
            {"id": "b", "title": "B", "content": "==z=="},
            {"id": "a", "title": "A", "content": "==a=="},
        ]
    )

    assert [(row["unit_id"], row["context"]) for row in rows(text)] == [("a", "==a=="), ("b", "==z==")]
