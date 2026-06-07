import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_mathml_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_mathml_csv_summarizes_math_and_skips_fences(tmp_path):
    content = """```
<math><mi>skip</mi></math>
```
<math display="block" alttext="Area &amp; total"><mi>a</mi><mo>+</mo><mi>b</mi><annotation encoding="application/x-tex">a+b</annotation></math>
<math><mrow><mi>x</mi><mo>=</mo><span>bad html</span></mrow></math>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_mathml_csv(units)
    result = rows(text)

    assert len(result) == 2
    assert result[0]["display"] == "block"
    assert result[0]["alttext"] == "Area & total"
    assert result[0]["annotation_count"] == "1"
    assert result[0]["identifier_count"] == "2"
    assert result[0]["operator_count"] == "1"
    assert result[0]["text_preview"] == "a + b a+b"
    assert result[1]["nested_html_present"] == "true"

    output = tmp_path / "mathml.csv"
    stats = export_units_to_markdown_html_mathml_csv(units, output)
    assert stats["rows_exported"] == 2
    assert output.read_text() == text
