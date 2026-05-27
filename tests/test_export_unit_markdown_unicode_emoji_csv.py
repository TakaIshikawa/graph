from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_unicode_emoji_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_unit_markdown_unicode_emoji_csv_groups_repeated_emoji():
    rows = _rows(export_unit_markdown_unicode_emoji_csv([{"id": "u1", "title": "Mood", "content": "Done ✅ ✅"}]))

    assert rows == [
        {
            "unit_id": "u1",
            "title": "Mood",
            "emoji": "✅",
            "count": "2",
            "first_line_number": "1",
            "contexts": "Done ✅ ✅",
        }
    ]


def test_export_unit_markdown_unicode_emoji_csv_tracks_multiple_lines_deterministically():
    rows = _rows(
        export_unit_markdown_unicode_emoji_csv(
            [{"id": "u1", "content": "First 🚀\nSecond 🎯 🚀\nThird 🎯"}]
        )
    )

    assert [(row["emoji"], row["count"], row["first_line_number"], row["contexts"]) for row in rows] == [
        ("🚀", "2", "1", "First 🚀; Second 🎯 🚀"),
        ("🎯", "2", "2", "Second 🎯 🚀; Third 🎯"),
    ]


def test_export_unit_markdown_unicode_emoji_csv_ignores_colon_shortcodes_and_writes_path(tmp_path):
    path = tmp_path / "emoji.csv"
    units = [{"id": "u1", "content": ":rocket:\nActual 🚀"}]
    expected = export_unit_markdown_unicode_emoji_csv(units)

    stats = export_unit_markdown_unicode_emoji_csv(units, path)

    assert [row["emoji"] for row in _rows(expected)] == ["🚀"]
    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": len(expected.encode("utf-8")),
    }
