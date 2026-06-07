import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_svg_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_svg_csv_exports_accessibility_structure_and_skips_fences(tmp_path):
    content = """```
<svg><title>Skip</title></svg>
```
<svg id="icon" class="glyph" width="24" height="24" viewBox="0 0 24 24" role="img" aria-label="Open &amp; close">
  <title>Door &amp; frame</title><desc>Line icon</desc><use href="https://cdn.example.com/icons.svg#door"></use><path d="M0 0"/>
</svg>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_svg_csv(units)
    result = rows(text)

    assert len(result) == 1
    row = result[0]
    assert row["id"] == "icon"
    assert row["class"] == "glyph"
    assert row["width"] == "24"
    assert row["height"] == "24"
    assert row["viewbox"] == "0 0 24 24"
    assert row["role"] == "img"
    assert row["aria_label"] == "Open & close"
    assert row["title_text"] == "Door & frame"
    assert row["desc_text"] == "Line icon"
    assert row["child_element_count"] == "4"
    assert row["has_external_reference"] == "true"

    output = tmp_path / "svg.csv"
    stats = export_units_to_markdown_html_svg_csv(units, output)
    assert stats["rows_exported"] == 1
    assert output.read_text() == text
