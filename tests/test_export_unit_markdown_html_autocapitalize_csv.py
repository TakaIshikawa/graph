import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_autocapitalize_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_autocapitalize_csv_classifies_values_previews_and_skips_fences():
    content = """```
<input autocapitalize="words">
```
<input id="off" autocapitalize="off">
<textarea autocapitalize="sentences">hello world</textarea>
<div autocapitalize="characters"><span>abc</span></div>
<p autocapitalize="none">none</p>"""

    result = rows(export_units_to_markdown_html_autocapitalize_csv([{"id": "u", "content": content}]))

    assert [row["normalized_value"] for row in result] == ["off", "sentences", "characters", "none"]
    assert result[0]["is_off"] == "true"
    assert result[0]["id"] == "off"
    assert result[1]["is_sentences"] == "true"
    assert result[1]["text_preview"] == "hello world"
    assert result[2]["is_characters"] == "true"
    assert result[2]["text_preview"] == "abc"
    assert result[3]["is_none"] == "true"
