import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_button_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_button_csv_exports_controls_without_value_text_and_skips_fences():
    content = """```html
<button>Skip</button>
```
<button id="save" name="action" type="submit" value="secret" formaction="/save" formmethod="post" formtarget="_blank"><span>Save</span> now</button>
<button type="reset" disabled>Reset</button>
<button type="button">Plain</button>"""

    result = rows(export_units_to_markdown_html_button_csv([{"id": "u", "content": content}]))

    assert [row["type"] for row in result] == ["submit", "reset", "button"]
    assert result[0]["value_present"] == "true"
    assert result[0]["formaction"] == "/save"
    assert result[0]["formmethod"] == "post"
    assert result[0]["formtarget"] == "_blank"
    assert result[0]["text_preview"] == "Save now"
    assert result[0]["has_html_content"] == "true"
    assert result[1]["disabled"] == "true"
