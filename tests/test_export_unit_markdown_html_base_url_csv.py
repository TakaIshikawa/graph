from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_base_url_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_base_url_csv_exports_attrs_and_skips_fences():
    content = """```
<base href="skip/" target="_blank">
```
<base id="b" class="root" href="https://example.com/docs/" target="_self">
<base target="_blank">
<base href="/relative/">"""

    rows = _rows(export_units_to_markdown_html_base_url_csv([{"id": "u", "title": "T", "source_path": "doc.md", "source": "s", "content": content}]))

    assert [row["href"] for row in rows] == ["https://example.com/docs/", "", "/relative/"]
    assert rows[0]["target"] == "_self"
    assert rows[0]["has_href"] == "true"
    assert rows[0]["has_target"] == "true"
    assert rows[0]["unsafe_blank_target"] == "false"
    assert rows[0]["id"] == "b"
    assert rows[1]["has_href"] == "false"
    assert rows[1]["has_target"] == "true"
    assert rows[1]["unsafe_blank_target"] == "true"
    assert rows[2]["has_target"] == "false"


def test_base_url_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "base.csv"
    units = [{"id": "u", "content": '<base href="/" target="_blank">'}]

    expected = export_units_to_markdown_html_base_url_csv(units)
    stats = export_units_to_markdown_html_base_url_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
