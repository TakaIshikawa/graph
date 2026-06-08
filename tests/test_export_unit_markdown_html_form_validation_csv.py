from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_form_validation_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_form_validation_attributes_and_skips_fences():
    content = """```
<input required>
```
<form id="f" novalidate>
<input id="email" class="wide" type="email" name="email" required pattern=".+@.+" minlength="3" maxlength="80">
<input type="number" name="qty" min="1" max="9" step="1">
<textarea name="bio" required maxlength="120"></textarea>
<select name="plan" required></select>
</form>"""

    rows = _rows(export_units_to_markdown_html_form_validation_csv([{"id": "u", "content": content}]))

    assert [row["tag"] for row in rows] == ["form", "input", "input", "textarea", "select"]
    assert rows[0]["novalidate"] == "true"
    assert rows[1]["required"] == "true"
    assert rows[1]["pattern"] == ".+@.+"
    assert rows[1]["minlength"] == "3"
    assert rows[1]["id"] == "email"
    assert rows[2]["min"] == "1"
    assert rows[2]["max"] == "9"
    assert rows[2]["step"] == "1"
    assert rows[3]["maxlength"] == "120"
    assert rows[4]["required"] == "true"
