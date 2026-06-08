import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_meter_progress_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_meter_progress_csv_exports_attributes_preview_sorting_and_skips_fences():
    content = """```
<meter value="1">Skip</meter>
```
<progress id="load">Loading</progress>
<meter id="score" min="0" max="10" value="7" low="2" high="8" optimum="6"><strong>Seven</strong></meter>"""

    result = rows(export_units_to_markdown_html_meter_progress_csv([{"id": "u", "content": content}]))

    assert [row["tag_name"] for row in result] == ["progress", "meter"]
    assert result[0]["has_value"] == "false"
    assert result[0]["value"] == ""
    assert result[0]["text_preview"] == "Loading"
    assert result[1]["value"] == "7"
    assert result[1]["min"] == "0"
    assert result[1]["max"] == "10"
    assert result[1]["low"] == "2"
    assert result[1]["high"] == "8"
    assert result[1]["optimum"] == "6"
    assert result[1]["has_value"] == "true"
    assert result[1]["text_preview"] == "Seven"
