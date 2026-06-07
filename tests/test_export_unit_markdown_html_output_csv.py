import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_output_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_output_csv_exports_form_accessibility_metadata_and_skips_fences(tmp_path):
    content = """```
<output name="skip">Skip</output>
```
<output id="total" class="result" name="sum" for="a b" form="calc" aria-live="polite" role="status">Total &amp; fees</output>
<output name="rich"><strong>42</strong></output>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_output_csv(units)
    result = rows(text)

    assert [row["name"] for row in result] == ["sum", "rich"]
    assert result[0]["for"] == "a b"
    assert result[0]["form"] == "calc"
    assert result[0]["text_preview"] == "Total & fees"
    assert result[0]["has_value_text"] == "true"
    assert result[0]["aria_live"] == "polite"
    assert result[0]["role"] == "status"
    assert result[0]["class"] == "result"
    assert result[0]["id"] == "total"
    assert result[1]["nested_html_present"] == "true"

    output = tmp_path / "output.csv"
    stats = export_units_to_markdown_html_output_csv(units, output)
    assert stats["rows_exported"] == 2
    assert output.read_text() == text
