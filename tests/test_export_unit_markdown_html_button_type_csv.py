import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_button_type_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_button_type_csv_classifies_defaults_invalid_disabled_and_skips_fences():
    content = """```html
<button type="button">Skip</button>
```
<button id="save" name="action" value="go" form="f"><span>Save</span> now</button>
<button type="button">Plain</button>
<button type="reset" disabled>Reset</button>
<button type="menu">Invalid</button>"""

    result = rows(export_units_to_markdown_html_button_type_csv([{"id": "u", "content": content}]))

    assert [row["normalized_type"] for row in result] == ["submit", "button", "reset", "submit"]
    assert result[0]["type"] == ""
    assert result[0]["is_submit"] == "true"
    assert result[0]["name"] == "action"
    assert result[0]["value"] == "go"
    assert result[0]["form"] == "f"
    assert result[0]["text_preview"] == "Save now"
    assert result[1]["is_button"] == "true"
    assert result[2]["disabled"] == "true"
    assert result[3]["type"] == "menu"
