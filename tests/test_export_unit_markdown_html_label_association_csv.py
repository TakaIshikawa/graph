from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_label_association_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_label_association_csv_exports_variants_and_skips_fences():
    content = """```
<label for="skip">Skip</label>
```
<label id="l1" class="field" for="email">Email</label>
<label>Agree <input type="checkbox"></label>
<label>Loose</label>"""

    rows = _rows(export_units_to_markdown_html_label_association_csv([{"id": "u", "content": content}]))

    assert [row["text_preview"] for row in rows] == ["Email", "Agree", "Loose"]
    assert rows[0]["for_attr"] == "email"
    assert rows[0]["has_for"] == "true"
    assert rows[0]["wraps_control"] == "false"
    assert rows[0]["id"] == "l1"
    assert rows[1]["has_for"] == "false"
    assert rows[1]["wraps_control"] == "true"
    assert rows[2]["wraps_control"] == "false"


def test_label_association_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "labels.csv"
    units = [{"id": "u", "content": '<label for="x">X</label>'}]

    expected = export_units_to_markdown_html_label_association_csv(units)
    stats = export_units_to_markdown_html_label_association_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
