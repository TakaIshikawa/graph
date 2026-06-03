from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_admonition_block_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_callout_and_colon_fenced_admonition_starts():
    content = "> [!NOTE] Read this\n> body\n::: warning Be careful\ntext"
    result = rows(export_units_to_markdown_admonition_block_csv([{"id": "u", "title": "Doc", "content": content}]))

    assert [(row["admonition_type"], row["marker_style"], row["line_number"], row["title_text"]) for row in result] == [
        ("note", "blockquote_callout", "1", "Read this"),
        ("warning", "colon_fence", "3", "Be careful"),
    ]
