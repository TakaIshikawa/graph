from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_fieldset_legend_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_fieldset_legend_csv_exports_legend_and_control_counts():
    content = """```
<fieldset><legend>Skip</legend><input></fieldset>
```
<fieldset id="prefs" class="group" name="settings" disabled>
  <legend>Preferences &amp; alerts</legend>
  <input name="email">
  <select><option>Daily</option></select>
  <textarea></textarea>
  <button>Save</button>
</fieldset>"""

    result = _rows(export_units_to_markdown_html_fieldset_legend_csv([{"id": "u", "content": content}]))

    assert len(result) == 1
    assert result[0]["fieldset_id"] == "prefs"
    assert result[0]["fieldset_class"] == "group"
    assert result[0]["disabled"] == "true"
    assert result[0]["name"] == "settings"
    assert result[0]["legend_text"] == "Preferences & alerts"
    assert result[0]["control_count"] == "4"
    assert result[0]["input_count"] == "1"
    assert result[0]["select_count"] == "1"
    assert result[0]["textarea_count"] == "1"
