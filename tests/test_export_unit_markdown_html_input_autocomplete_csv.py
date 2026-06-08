from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_input_autocomplete_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_autocomplete_tokens_inheritance_and_skips_fences():
    content = """```
<input autocomplete="skip">
```
<form autocomplete="off">
<input type="email" name="email" autocomplete="section-blue shipping email">
<textarea name="bio"></textarea>
<select name="country" autocomplete="country"></select>
</form>"""

    rows = _rows(export_units_to_markdown_html_input_autocomplete_csv([{"id": "u", "content": content}]))

    assert [row["tag"] for row in rows] == ["form", "input", "textarea", "select"]
    assert rows[0]["autocomplete"] == "off"
    assert rows[0]["disables_autocomplete"] == "true"
    assert rows[1]["autocomplete_tokens"] == "3"
    assert rows[1]["has_section_token"] == "true"
    assert rows[1]["disables_autocomplete"] == "false"
    assert rows[2]["autocomplete"] == "off"
    assert rows[2]["disables_autocomplete"] == "true"
    assert rows[3]["autocomplete"] == "country"
