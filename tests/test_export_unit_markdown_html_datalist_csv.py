import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_datalist_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_datalist_csv_exports_summary_options_empty_lists_and_skips_fences():
    content = """```html
<datalist id="skip"><option value="x"></option></datalist>
```
<datalist id="cities" class="geo"><option value="tokyo" label="Tokyo"></option><option value="nyc">New York</option></datalist>
<datalist id="empty"></datalist>"""

    result = rows(export_units_to_markdown_html_datalist_csv([{"id": "u", "content": content}]))

    assert [row["row_type"] for row in result] == ["datalist", "option", "option", "datalist"]
    assert result[0]["datalist_id"] == "cities"
    assert result[0]["option_count"] == "2"
    assert result[0]["class"] == "geo"
    assert result[1]["option_value"] == "nyc"
    assert result[1]["option_text"] == "New York"
    assert result[2]["option_value"] == "tokyo"
    assert result[2]["option_label"] == "Tokyo"
    assert result[3]["datalist_id"] == "empty"
    assert result[3]["option_count"] == "0"
