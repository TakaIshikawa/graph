from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_table_caption_colgroup_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_caption_colgroup_col_and_skips_fences():
    content = """```
<caption>Skip</caption>
```
<table><caption id="cap" class="small">Main <b>table</b></caption><colgroup id="cg"><col span="2"><col></colgroup><caption></caption><colgroup span="3"></colgroup></table>"""

    rows = _rows(export_units_to_markdown_html_table_caption_colgroup_csv([{"id": "u", "content": content}]))

    assert [row["tag"] for row in rows] == ["caption", "colgroup", "col", "col", "caption", "colgroup"]
    assert rows[0]["text_preview"] == "Main table"
    assert rows[0]["empty_caption"] == "false"
    assert rows[0]["id"] == "cap"
    assert rows[1]["column_count"] == "2"
    assert rows[2]["span"] == "2"
    assert rows[2]["column_count"] == "2"
    assert rows[4]["empty_caption"] == "true"
    assert rows[5]["span"] == "3"
    assert rows[5]["column_count"] == "3"
