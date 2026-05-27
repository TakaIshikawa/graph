from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_heading_anchor_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_atx_heading_anchors():
    [row] = rows(export_units_to_markdown_heading_anchor_csv([{"id": "u", "content": "## Title Here {#custom-id}"}]))
    assert row["level"] == "2"
    assert row["heading_text"] == "Title Here"
    assert row["anchor_id"] == "custom-id"


def test_ignores_non_heading_brace_fragments():
    assert rows(export_units_to_markdown_heading_anchor_csv([{"id": "u", "content": "Text {#custom-id}\n## Missing anchor"}])) == []
