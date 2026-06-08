from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_heading_outline_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_heading_outline_jumps_empty_and_skips_fences():
    content = """```
<h1>Skip</h1>
```
<h1 id="top" class="hero">Title</h1>
<h3>Deep</h3>
<h2><span></span></h2>"""

    rows = _rows(export_units_to_markdown_html_heading_outline_csv([{"id": "u", "content": content}]))

    assert [row["level"] for row in rows] == ["1", "3", "2"]
    assert rows[0]["id"] == "top"
    assert rows[0]["class"] == "hero"
    assert rows[1]["outline_jump_from_previous"] == "true"
    assert rows[2]["empty_heading"] == "true"
