from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_footnote_reference_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_footnote_reference_csv_exports_references_not_definitions():
    text = export_units_to_markdown_footnote_reference_csv(
        [{"id": "u", "title": "Doc", "content": "Text[^a] again[^a] other[^b]\n[^a]: Definition\n```md\n[^ignored]\n```"}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Doc", "label": "a", "line_number": "1", "reference_count_on_line": "3", "source_url": ""},
        {"unit_id": "u", "title": "Doc", "label": "a", "line_number": "1", "reference_count_on_line": "3", "source_url": ""},
        {"unit_id": "u", "title": "Doc", "label": "b", "line_number": "1", "reference_count_on_line": "3", "source_url": ""},
    ]


def test_markdown_footnote_reference_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "footnotes.csv"
    units = [{"id": "u", "content": "[^x]"}]

    expected = export_units_to_markdown_footnote_reference_csv(units)
    stats = export_units_to_markdown_footnote_reference_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
