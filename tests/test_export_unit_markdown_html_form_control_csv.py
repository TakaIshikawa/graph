import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_form_control_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_form_control_csv_exports_controls_booleans_and_line_numbers():
    content = """```html
<input name="skip">
```
<input id="email" name="email" type="email" value="secret" placeholder="Email" required autocomplete="email" checked form="f">
<select id="plan" name="plan" multiple disabled></select>
<textarea id="bio" name="bio" rows="4" cols="20"></textarea>"""

    result = rows(export_units_to_markdown_html_form_control_csv([{"id": "u", "content": content}]))

    assert [row["tag"] for row in result] == ["input", "select", "textarea"]
    assert result[0]["line_number"] == "4"
    assert result[0]["value_present"] == "true"
    assert result[0]["placeholder"] == "Email"
    assert result[0]["required"] == "true"
    assert result[0]["checked"] == "true"
    assert result[1]["multiple"] == "true"
    assert result[1]["disabled"] == "true"
    assert result[2]["rows"] == "4"
    assert result[2]["cols"] == "20"
