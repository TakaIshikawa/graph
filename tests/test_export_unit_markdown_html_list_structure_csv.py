from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_list_structure_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_list_structure_nested_items_and_skips_fences():
    content = """```
<ul><li>Skip</li></ul>
```
<ol id="steps" class="flow" start="3" reversed><li>One<ul><li>Nested</li></ul></li><li>Two</li></ol>"""

    rows = _rows(export_units_to_markdown_html_list_structure_csv([{"id": "u", "content": content}]))

    assert [row["tag"] for row in rows] == ["ol", "li", "ul", "li", "li"]
    assert rows[0]["list_type"] == "ordered"
    assert rows[0]["start"] == "3"
    assert rows[0]["reversed"] == "true"
    assert rows[0]["item_count"] == "3"
    assert rows[0]["id"] == "steps"
    assert rows[1]["text_preview"] == "One Nested"
    assert rows[3]["nesting_depth"] == "2"
