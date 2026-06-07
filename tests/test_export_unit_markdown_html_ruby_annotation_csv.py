import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_ruby_annotation_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_ruby_annotation_csv_exports_annotations_and_skips_fences(tmp_path):
    content = """```
<ruby>悪<rt>skip</rt></ruby>
```
<ruby>漢<rp>(</rp><rt>kan</rt><rp>)</rp>字<rt>ji</rt></ruby>
<ruby><span>語</span><rt>go</rt></ruby>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_ruby_annotation_csv(units)
    result = rows(text)

    assert len(result) == 2
    assert result[0]["ruby_text"] == "漢 字"
    assert result[0]["rt_text"] == "kan | ji"
    assert result[0]["rp_text"] == "( | )"
    assert result[0]["annotation_count"] == "2"
    assert result[0]["has_fallback_parentheses"] == "true"
    assert result[1]["nested_html_present"] == "true"

    output = tmp_path / "ruby.csv"
    stats = export_units_to_markdown_html_ruby_annotation_csv(units, output)
    assert stats["rows_exported"] == 2
    assert output.read_text() == text
