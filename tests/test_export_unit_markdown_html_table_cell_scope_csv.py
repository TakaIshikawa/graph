import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_table_cell_scope_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_table_cell_scope_csv_exports_accessibility_cells_and_skips_fences():
    content = """```
<th scope="col">Skip</th>
```
<table>
<tr><th abbr="Amt">Amount</th><th scope="col" colspan="2">Total</th></tr>
<tr><td headers="h1 h2" rowspan="2"><span>$10</span></td><td>ignored</td></tr>
</table>"""

    result = rows(export_units_to_markdown_html_table_cell_scope_csv([{"id": "u", "content": content}]))

    assert [(row["tag"], row["text_preview"]) for row in result] == [("th", "Amount"), ("th", "Total"), ("td", "$10")]
    assert result[0]["abbr"] == "Amt"
    assert result[0]["has_scope_or_headers"] == "false"
    assert result[1]["scope"] == "col"
    assert result[1]["colspan"] == "2"
    assert result[2]["headers"] == "h1 h2"
    assert result[2]["rowspan"] == "2"
