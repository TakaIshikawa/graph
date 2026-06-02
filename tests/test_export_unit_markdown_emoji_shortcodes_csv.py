from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_emoji_shortcodes_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_markdown_emoji_shortcodes_with_lines_and_context():
    result = rows(
        export_units_to_markdown_emoji_shortcodes_csv(
            [{"id": "u", "title": "T", "content": "Plan :memo:\nDone :white_check_mark: today"}]
        )
    )

    assert [(row["unit_id"], row["title"], row["shortcode"], row["line_number"]) for row in result] == [
        ("u", "T", ":memo:", "1"),
        ("u", "T", ":white_check_mark:", "2"),
    ]
    assert result[0]["context"] == "Plan :memo:"


def test_ignores_code_spans_and_fenced_code_examples():
    result = rows(
        export_units_to_markdown_emoji_shortcodes_csv(
            [
                {
                    "id": "u",
                    "content": "`:memo:` real :sparkles:\n```md\n:white_check_mark:\n```\nAfter :ok_hand:",
                }
            ]
        )
    )

    assert [(row["shortcode"], row["line_number"]) for row in result] == [
        (":sparkles:", "1"),
        (":ok_hand:", "5"),
    ]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "emoji.csv"

    result = export_units_to_markdown_emoji_shortcodes_csv([{"id": "u", "content": ":memo:"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
