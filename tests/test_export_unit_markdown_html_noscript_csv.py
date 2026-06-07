import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_noscript_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_noscript_csv_detects_fallback_elements_and_skips_fences(tmp_path):
    content = """```
<noscript><a href="/skip">Skip</a></noscript>
```
<noscript><a href="/help">Help &amp; docs</a><img src="x.png"><form></form></noscript>
<noscript>   </noscript>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_noscript_csv(units)
    result = rows(text)

    assert len(result) == 2
    assert result[0]["content_preview"] == "Help & docs"
    assert result[0]["contains_link"] == "true"
    assert result[0]["contains_image"] == "true"
    assert result[0]["contains_form"] == "true"
    assert result[0]["nested_tag_count"] == "3"
    assert result[1]["empty_content"] == "true"

    output = tmp_path / "noscript.csv"
    stats = export_units_to_markdown_html_noscript_csv(units, output)
    assert stats["rows_exported"] == 2
    assert output.read_text() == text
