from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_line_break_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_line_break_csv_exports_break_tags_attributes_and_context():
    content = """Before<br id="b1" class="soft">after
<hr role="separator" aria-hidden="true">
soft<wbr>wrap
```
<br id="skip">
```"""

    result = _rows(export_units_to_markdown_html_line_break_csv([{"id": "u", "content": content}]))

    assert [(row["tag"], row["line_number"]) for row in result] == [("br", "1"), ("hr", "2"), ("wbr", "3")]
    assert result[0]["id"] == "b1"
    assert result[0]["class"] == "soft"
    assert result[0]["surrounding_text_preview"].startswith("Before")
    assert result[1]["role"] == "separator"
    assert result[1]["aria_hidden"] == "true"
