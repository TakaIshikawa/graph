from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_edit_mark_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edit_mark_csv_exports_del_ins_metadata_and_links():
    content = """<ins cite="/change" datetime="2026-01-02">Added <a href="/r">reference</a> text</ins>
```
<del>skip</del>
```
<del>Removed &amp; archived</del>"""

    result = _rows(export_units_to_markdown_html_edit_mark_csv([{"id": "u", "content": content}]))

    assert [(row["tag"], row["line_number"]) for row in result] == [("ins", "1"), ("del", "5")]
    assert result[0]["cite"] == "/change"
    assert result[0]["datetime"] == "2026-01-02"
    assert result[0]["has_datetime"] == "true"
    assert result[0]["text_preview"] == "Added reference text"
    assert result[0]["word_count"] == "3"
    assert result[0]["link_count"] == "1"
    assert result[1]["cite"] == ""
    assert result[1]["datetime"] == ""
    assert result[1]["has_datetime"] == "false"
    assert result[1]["text_preview"] == "Removed & archived"
