import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_script_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_script_csv_exports_loading_security_and_inline_metadata(tmp_path):
    content = """```
<script src="https://skip.example/app.js"></script>
```
<script src="https://cdn.example.com/app.js" type="module" async defer nomodule integrity="sha384-x" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
<script>const label = "A &amp; B";</script>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_script_csv(units)
    result = rows(text)

    assert len(result) == 2
    assert result[0]["src"] == "https://cdn.example.com/app.js"
    assert result[0]["domain"] == "cdn.example.com"
    assert result[0]["type"] == "module"
    assert result[0]["async"] == "true"
    assert result[0]["defer"] == "true"
    assert result[0]["module"] == "true"
    assert result[0]["nomodule"] == "true"
    assert result[0]["integrity"] == "sha384-x"
    assert result[0]["crossorigin"] == "anonymous"
    assert result[0]["referrerpolicy"] == "no-referrer"
    assert result[1]["inline"] == "true"
    assert result[1]["inline_preview"] == 'const label = "A & B";'

    output = tmp_path / "script.csv"
    stats = export_units_to_markdown_html_script_csv(units, output)
    assert stats["rows_exported"] == 2
    assert output.read_text() == text
