from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_code_fence_line_counts_to_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_code_fence_line_count_csv_counts_backtick_tilde_and_unterminated_blocks():
    text = export_unit_markdown_code_fence_line_counts_to_csv(
        [{"id": "u", "title": "Unit", "content": "``` python\nx\n\n```\n~~~js\n\ncode\n~~~\n```\nopen"}]
    )

    assert _rows(text) == [
        {"unit_id": "u", "title": "Unit", "opening_line": "1", "closing_line": "4", "info_string": "python", "content_line_count": "2", "blank_line_count": "1", "unterminated": "false"},
        {"unit_id": "u", "title": "Unit", "opening_line": "5", "closing_line": "8", "info_string": "js", "content_line_count": "2", "blank_line_count": "1", "unterminated": "false"},
        {"unit_id": "u", "title": "Unit", "opening_line": "9", "closing_line": "", "info_string": "", "content_line_count": "1", "blank_line_count": "0", "unterminated": "true"},
    ]


def test_code_fence_line_count_csv_path_mode_writes_stats(tmp_path):
    path = tmp_path / "fences.csv"
    units = [{"id": "u", "content": "```\nx\n```"}]

    expected = export_unit_markdown_code_fence_line_counts_to_csv(units)
    stats = export_unit_markdown_code_fence_line_counts_to_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
