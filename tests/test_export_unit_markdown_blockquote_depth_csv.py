from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_blockquote_depth_csv


def _rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_blockquote_depth_csv_exports_single_nested_and_blank_quote_lines():
    text = export_units_to_markdown_blockquote_depth_csv(
        [
            {"id": "b", "title": "Beta", "source_project": "notes", "content": "> one\n>\n>> two"},
            {"id": "a", "title": "Alpha", "source": "docs", "content": "plain\n>>> three"},
        ]
    )

    assert _rows(text) == [
        {"unit_id": "a", "title": "Alpha", "source": "docs", "line_number": "2", "depth": "3", "quoted_text": "three", "is_blank_quote": "false"},
        {"unit_id": "b", "title": "Beta", "source": "notes", "line_number": "1", "depth": "1", "quoted_text": "one", "is_blank_quote": "false"},
        {"unit_id": "b", "title": "Beta", "source": "notes", "line_number": "2", "depth": "1", "quoted_text": "", "is_blank_quote": "true"},
        {"unit_id": "b", "title": "Beta", "source": "notes", "line_number": "3", "depth": "2", "quoted_text": "two", "is_blank_quote": "false"},
    ]


def test_markdown_blockquote_depth_csv_ignores_non_quotes_and_fences():
    content = "\n".join(["plain", "> kept", "```", ">> ignored", "```", "~~~", "> ignored too", "~~~"])

    rows = _rows(export_units_to_markdown_blockquote_depth_csv([{"id": "u", "content": content}]))

    assert rows == [
        {"unit_id": "u", "title": "", "source": "", "line_number": "2", "depth": "1", "quoted_text": "kept", "is_blank_quote": "false"}
    ]


def test_markdown_blockquote_depth_csv_path_mode_reports_write_metadata(tmp_path):
    path = tmp_path / "blockquote-depth.csv"
    units = [{"id": "u", "content": "> quote"}]

    expected = export_units_to_markdown_blockquote_depth_csv(units)
    stats = export_units_to_markdown_blockquote_depth_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
