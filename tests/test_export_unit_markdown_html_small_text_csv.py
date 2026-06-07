from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_small_text_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_small_text_csv_exports_legal_phrase_flags_links_and_write_stats(tmp_path):
    content = """```
<small>skip copyright</small>
```
<small id="legal" class="fine">Copyright &copy; 2026. <a href="/license">MIT License</a></small>"""
    units = [{"id": "u", "content": content}]

    text = export_units_to_markdown_html_small_text_csv(units)
    result = _rows(text)

    assert len(result) == 1
    assert result[0]["id"] == "legal"
    assert result[0]["class"] == "fine"
    assert result[0]["text_preview"] == "Copyright © 2026. MIT License"
    assert result[0]["word_count"] == "5"
    assert result[0]["link_count"] == "1"
    assert result[0]["contains_copyright"] == "true"
    assert result[0]["contains_license"] == "true"

    output = tmp_path / "small.csv"
    stats = export_units_to_markdown_html_small_text_csv(units, output)
    assert stats["rows_exported"] == 1
    assert output.read_text() == text
