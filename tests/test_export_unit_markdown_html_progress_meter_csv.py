import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_progress_meter_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_progress_meter_csv_normalizes_values_and_skips_fences():
    content = """```
<progress value="1"></progress>
```
<progress value="0.25">25%</progress>
<meter min="0" max="10" value="7" low="2" high="8" optimum="6"><strong>Seven</strong></meter>
<meter min="0" max="10" value="oops">Bad</meter>"""

    result = rows(export_units_to_markdown_html_progress_meter_csv([{"id": "u", "content": content}]))

    assert [row["tag"] for row in result] == ["progress", "meter", "meter"]
    assert result[0]["normalized_percent"] == "25"
    assert result[1]["normalized_percent"] == "70"
    assert result[1]["text_preview"] == "Seven"
    assert result[2]["normalized_percent"] == ""
