from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_time_datetime_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_time_datetime_csv_exports_preview_and_skips_fences():
    content = """```
<time datetime="skip">Skip</time>
```
<time id="t" class="when" datetime="2026-06-08">June <b>8</b></time>
<time>Later</time>"""

    rows = _rows(export_units_to_markdown_html_time_datetime_csv([{"id": "u", "content": content}]))

    assert [(row["datetime"], row["has_datetime"], row["text_preview"]) for row in rows] == [("2026-06-08", "true", "June 8"), ("", "false", "Later")]
    assert rows[0]["id"] == "t"


def test_time_datetime_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "times.csv"
    units = [{"id": "u", "content": "<time datetime=2026>Year</time>"}]

    expected = export_units_to_markdown_html_time_datetime_csv(units)
    stats = export_units_to_markdown_html_time_datetime_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
