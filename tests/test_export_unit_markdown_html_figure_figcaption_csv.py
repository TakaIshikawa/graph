from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_figure_figcaption_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_figure_figcaption_csv_exports_caption_metadata_and_skips_fences():
    content = """```
<figure><figcaption>Skip</figcaption></figure>
```
<figure id="f" class="media"><img src="a.png"><figcaption id="c" class="cap">A <b>caption</b></figcaption></figure>
<figure><img src="b.png"></figure>"""

    rows = _rows(export_units_to_markdown_html_figure_figcaption_csv([{"id": "u", "content": content}]))

    assert [row["has_figcaption"] for row in rows] == ["true", "false"]
    assert rows[0]["figure_id"] == "f"
    assert rows[0]["figure_class"] == "media"
    assert rows[0]["figcaption_count"] == "1"
    assert rows[0]["figcaption_text_preview"] == "A caption"
    assert rows[0]["figcaption_id"] == "c"
    assert rows[1]["figcaption_count"] == "0"


def test_figure_figcaption_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "figures.csv"
    units = [{"id": "u", "content": "<figure><figcaption>X</figcaption></figure>"}]

    expected = export_units_to_markdown_html_figure_figcaption_csv(units)
    stats = export_units_to_markdown_html_figure_figcaption_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
