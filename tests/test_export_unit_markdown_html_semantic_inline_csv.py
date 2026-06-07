from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_semantic_inline_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_semantic_inline_csv_exports_tags_title_and_empty_text():
    content = """<samp id="out">Ready &amp; set</samp>
<var class="symbol">x</var>
<dfn title="Hypertext Markup Language"></dfn>
```
<samp>skip</samp>
```"""

    result = _rows(export_units_to_markdown_html_semantic_inline_csv([{"id": "u", "content": content}]))

    assert [(row["tag"], row["line_number"]) for row in result] == [("samp", "1"), ("var", "2"), ("dfn", "3")]
    assert result[0]["id"] == "out"
    assert result[0]["text_preview"] == "Ready & set"
    assert result[0]["has_title"] == "false"
    assert result[1]["class"] == "symbol"
    assert result[2]["title"] == "Hypertext Markup Language"
    assert result[2]["has_title"] == "true"
    assert result[2]["empty_text"] == "true"
