from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_autolink_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_autolink_csv_classifies_urls_and_emails_not_html_tags():
    text = export_units_to_markdown_autolink_csv(
        [
            {"id": "b", "title": "Beta", "metadata": {"source_url": "https://src"}, "content": "<user@example.com> <span>\n```\n<https://ignored.test>\n```"},
            {"id": "a", "title": "Alpha", "content": "<https://example.com>"},
        ]
    )

    assert _rows(text) == [
        {"unit_id": "a", "title": "Alpha", "target": "https://example.com", "target_type": "url", "line_number": "1", "source_url": ""},
        {"unit_id": "b", "title": "Beta", "target": "user@example.com", "target_type": "email", "line_number": "1", "source_url": "https://src"},
    ]


def test_markdown_autolink_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "autolinks.csv"
    units = [{"id": "u", "content": "<mailto:user@example.com>"}]

    expected = export_units_to_markdown_autolink_csv(units)
    stats = export_units_to_markdown_autolink_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
