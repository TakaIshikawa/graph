from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_meta_charset_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_meta_charset_csv_exports_charset_declarations_and_skips_fences():
    content = """```
<meta charset="skip">
```
<meta id="m" class="head" charset="utf-8">
<meta http-equiv="Content-Type" content="text/html; charset=shift_jis">
<meta name="viewport" content="width=device-width">"""

    rows = _rows(export_units_to_markdown_html_meta_charset_csv([{"id": "u", "content": content}]))

    assert len(rows) == 2
    assert rows[0]["charset"] == "utf-8"
    assert rows[0]["declares_charset"] == "true"
    assert rows[0]["id"] == "m"
    assert rows[1]["http_equiv"] == "Content-Type"
    assert rows[1]["content"] == "text/html; charset=shift_jis"
    assert rows[1]["declares_charset"] == "true"


def test_meta_charset_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "meta.csv"
    units = [{"id": "u", "content": '<meta http-equiv="content-type" content="text/html">'}]

    expected = export_units_to_markdown_html_meta_charset_csv(units)
    stats = export_units_to_markdown_html_meta_charset_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
