from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_html_anchor_rel_security_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_anchor_rel_security_and_skips_fences(tmp_path):
    content = """```
<a href="skip" target="_blank">Skip</a>
```
<a id="bad" class="cta" href="https://example.test" target="_blank"><strong>Open</strong> site</a>
<a href="/safe" target="_blank" rel="NoOpener Noreferrer">Safe</a>
<a href="/normal">Normal</a>"""
    units = [{"id": "u", "title": "Unit", "source_path": "u.md", "content": content}]

    rows = _rows(export_units_to_markdown_html_anchor_rel_security_csv(units))

    assert [row["href"] for row in rows] == ["https://example.test", "/safe", "/normal"]
    assert rows[0]["line_number"] == "4"
    assert rows[0]["opens_new_context"] == "true"
    assert rows[0]["unsafe_blank_target"] == "true"
    assert rows[0]["has_noopener"] == "false"
    assert rows[0]["text_preview"] == "Open site"
    assert rows[0]["id"] == "bad"
    assert rows[1]["has_noopener"] == "true"
    assert rows[1]["has_noreferrer"] == "true"
    assert rows[1]["unsafe_blank_target"] == "false"
    assert rows[2]["opens_new_context"] == "false"

    output = tmp_path / "anchors.csv"
    result = export_units_to_markdown_html_anchor_rel_security_csv(units, output)
    assert result["rows_exported"] == 3
    assert output.read_text() == export_units_to_markdown_html_anchor_rel_security_csv(units)
