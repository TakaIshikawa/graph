import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_canvas_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_canvas_csv_exports_fallback_metadata_and_skips_fences(tmp_path):
    content = """```
<canvas id="skip">Skip</canvas>
```
<canvas id="plot" class="chart" width="640" height="320" aria-label="Revenue &amp; costs" role="img">
  <p>Fallback <strong>chart</strong> &amp; details</p>
</canvas>
<canvas id="empty"></canvas>"""
    units = [{"id": "u", "title": "Canvas", "source_path": "doc.md", "source": "manual", "content": content}]

    text = export_units_to_markdown_html_canvas_csv(units)
    result = rows(text)

    assert [row["id"] for row in result] == ["plot", "empty"]
    assert result[0]["class"] == "chart"
    assert result[0]["width"] == "640"
    assert result[0]["height"] == "320"
    assert result[0]["aria_label"] == "Revenue & costs"
    assert result[0]["role"] == "img"
    assert result[0]["fallback_preview"] == "Fallback chart & details"
    assert result[0]["has_fallback_content"] == "true"
    assert result[0]["nested_html_present"] == "true"
    assert result[1]["width"] == ""
    assert result[1]["height"] == ""
    assert result[1]["has_fallback_content"] == "false"

    output = tmp_path / "canvas.csv"
    stats = export_units_to_markdown_html_canvas_csv(units, output)
    assert stats["rows_exported"] == 2
    assert output.read_text() == text
