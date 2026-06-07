from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_table_structure_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_table_structure_csv_exports_caption_sections_and_rows():
    content = """```
<table><tr><td>skip</td></tr></table>
```
<table>
  <caption>Revenue &amp; costs</caption>
  <colgroup><col><col></colgroup>
  <thead><tr><th>A</th></tr></thead>
  <tbody><tr><td>1</td></tr><tr><td>2</td></tr></tbody>
  <tfoot><tr><td>Total</td></tr></tfoot>
</table>"""

    result = _rows(export_units_to_markdown_html_table_structure_csv([{"id": "u", "content": content}]))

    assert len(result) == 1
    assert result[0]["caption_text"] == "Revenue & costs"
    assert result[0]["colgroup_count"] == "1"
    assert result[0]["col_count"] == "2"
    assert result[0]["thead_count"] == "1"
    assert result[0]["tbody_count"] == "1"
    assert result[0]["tfoot_count"] == "1"
    assert result[0]["row_count"] == "4"
