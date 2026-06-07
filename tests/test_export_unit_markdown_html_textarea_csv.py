from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_textarea_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_textarea_csv_exports_attributes_value_and_write_stats(tmp_path):
    content = """```
<textarea name="skip">ignored</textarea>
```
<textarea name="note" id="n" rows="4" cols="30" maxlength="200" required readonly placeholder="Tell &amp; show">Hello &amp; goodbye</textarea>
<textarea name="empty" disabled></textarea>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_textarea_csv(units)
    result = _rows(text)

    assert [row["name"] for row in result] == ["note", "empty"]
    assert result[0]["id"] == "n"
    assert result[0]["rows"] == "4"
    assert result[0]["cols"] == "30"
    assert result[0]["maxlength"] == "200"
    assert result[0]["required"] == "true"
    assert result[0]["readonly"] == "true"
    assert result[0]["disabled"] == "false"
    assert result[0]["placeholder"] == "Tell & show"
    assert result[0]["value_preview"] == "Hello & goodbye"
    assert result[0]["empty_value"] == "false"
    assert result[1]["disabled"] == "true"
    assert result[1]["empty_value"] == "true"

    output = tmp_path / "textarea.csv"
    stats = export_units_to_markdown_html_textarea_csv(units, output)
    assert stats["rows_exported"] == 2
    assert output.read_text() == text
