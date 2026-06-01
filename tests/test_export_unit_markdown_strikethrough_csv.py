from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_strikethrough_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_strikethrough_csv_exports_double_tilde_spans_only():
    text = export_units_to_markdown_strikethrough_csv(
        [{"id": "u", "title": "Doc", "source_url": "src", "content": "Keep ~single~ and ~~one~~ plus ~~two~~\n```md\n~~ignored~~\n```"}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Doc", "text": "one", "line_number": "1", "span_start": "19", "source_url": "src"},
        {"unit_id": "u", "title": "Doc", "text": "two", "line_number": "1", "span_start": "32", "source_url": "src"},
    ]


def test_markdown_strikethrough_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "strike.csv"
    units = [{"id": "u", "content": "~~gone~~"}]

    expected = export_units_to_markdown_strikethrough_csv(units)
    stats = export_units_to_markdown_strikethrough_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
