import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_dialog_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_dialog_csv_exports_accessibility_preview_multiline_and_skips_fences():
    content = """```
<dialog open>Skip</dialog>
```
<dialog id="confirm" open aria-label="Confirm &amp; continue" aria-labelledby="title" role="alertdialog">
<h2>Confirm</h2>
Continue?
</dialog>"""

    result = rows(export_units_to_markdown_html_dialog_csv([{"id": "u", "content": content}]))

    assert len(result) == 1
    assert result[0]["id"] == "confirm"
    assert result[0]["open"] == "true"
    assert result[0]["aria_label"] == "Confirm & continue"
    assert result[0]["aria_labelledby"] == "title"
    assert result[0]["role"] == "alertdialog"
    assert result[0]["text_preview"] == "Confirm Continue?"
    assert result[0]["multiline"] == "true"
