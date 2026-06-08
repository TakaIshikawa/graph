from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_details_summary_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_details_summary_variants_and_skips_fences():
    content = """```
<details><summary>Skip</summary></details>
```
<details id="d" class="box" open><summary id="s">First <b>summary</b></summary><summary>Second</summary><p>Body</p></details>
<details>Body only</details>"""

    rows = _rows(export_units_to_markdown_html_details_summary_csv([{"id": "u", "content": content}]))

    assert [row["tag"] for row in rows] == ["details", "summary", "summary", "details"]
    assert rows[0]["open"] == "true"
    assert rows[0]["summary_text_preview"] == "First summary"
    assert rows[0]["summary_count"] == "2"
    assert rows[0]["missing_summary"] == "false"
    assert rows[0]["id"] == "d"
    assert rows[1]["id"] == "s"
    assert rows[3]["missing_summary"] == "true"
    assert rows[3]["summary_count"] == "0"
