from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_time_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_time_csv_exports_datetime_and_text_only_tags():
    text = export_unit_markdown_html_time_csv(
        [{"id": "u", "title": "T", "source": "s", "content": '<time datetime="2026-06-04">Today</time>\n<time>Later</time>'}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "datetime": "2026-06-04", "text": "Today", "has_datetime": "True"},
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "2", "datetime": "", "text": "Later", "has_datetime": "False"},
    ]


def test_time_csv_exports_multiple_tags_per_line_and_is_importable():
    text = export_unit_markdown_html_time_csv([{"id": "u", "content": "<time>Morning</time> <time datetime='18:00'>Evening</time>"}])

    assert [(row["line_number"], row["datetime"], row["text"]) for row in _rows(text)] == [("1", "", "Morning"), ("1", "18:00", "Evening")]


def test_time_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "time.csv"
    units = [{"id": "u", "content": "<time datetime=2026>Year</time>"}]

    expected = export_unit_markdown_html_time_csv(units)
    stats = export_unit_markdown_html_time_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
