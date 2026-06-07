from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_select_option_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_select_option_csv_summarizes_select_options():
    content = """```
<select><option>Skip</option></select>
```
<select name="plan" id="plan" multiple required disabled>
  <option value="">Choose</option>
  <option value="pro" selected>Pro &amp; Team</option>
  <option selected label="Enterprise" value="enterprise"></option>
</select>"""

    result = _rows(export_units_to_markdown_html_select_option_csv([{"id": "u", "content": content}]))

    assert len(result) == 1
    assert result[0]["select_name"] == "plan"
    assert result[0]["select_id"] == "plan"
    assert result[0]["multiple"] == "true"
    assert result[0]["required"] == "true"
    assert result[0]["disabled"] == "true"
    assert result[0]["option_count"] == "3"
    assert result[0]["selected_count"] == "2"
    assert result[0]["empty_value_count"] == "1"
    assert result[0]["option_values"] == "|pro|enterprise"
    assert result[0]["selected_labels_preview"] == "Pro & Team; Enterprise"
