from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_word_break_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_word_break_csv_exports_multiple_forms_and_skips_fences():
    content = """```
alpha<wbr>skip
```
super<wbr id="a">califragilistic
line two long<wbr class="soft"/>break"""

    rows = _rows(export_units_to_markdown_html_word_break_csv([{"id": "u", "content": content}]))

    assert [row["line_number"] for row in rows] == ["4", "5"]
    assert rows[0]["before_text_preview"].endswith("super")
    assert rows[0]["after_text_preview"].startswith("califragilistic")
    assert rows[0]["surrounding_text_preview"] == "supercalifragilistic line two long break"
    assert rows[0]["id"] == "a"
    assert rows[1]["class"] == "soft"


def test_word_break_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "wbr.csv"
    units = [{"id": "u", "content": "a<wbr>b"}]

    expected = export_units_to_markdown_html_word_break_csv(units)
    stats = export_units_to_markdown_html_word_break_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
