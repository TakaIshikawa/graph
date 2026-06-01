from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_math_blocks_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_closed_markdown_math_blocks():
    text = export_unit_markdown_math_blocks_to_csv(
        [
            {
                "id": "u1",
                "title": "Derivation",
                "content": "Before\n$$\na = b\n\nc = d\n$$\nAfter",
            }
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "u1",
            "title": "Derivation",
            "opening_line": "2",
            "closing_line": "6",
            "expression_line_count": "3",
            "blank_line_count": "1",
            "unterminated": "false",
        }
    ]


def test_reports_unterminated_markdown_math_blocks():
    text = export_unit_markdown_math_blocks_to_csv([{"id": "u1", "content": "$$\nx\n\nz"}])

    assert rows(text) == [
        {
            "unit_id": "u1",
            "title": "",
            "opening_line": "1",
            "closing_line": "",
            "expression_line_count": "3",
            "blank_line_count": "1",
            "unterminated": "true",
        }
    ]


def test_math_block_export_has_deterministic_ordering():
    text = export_unit_markdown_math_blocks_to_csv(
        [
            {"id": "b", "content": "$$\nb1\n$$\n$$\nb2\n$$"},
            {"id": "a", "metadata": {"title": "Alpha"}, "content": "Intro\n$$\na\n$$"},
        ]
    )

    assert rows(text) == [
        {
            "unit_id": "a",
            "title": "Alpha",
            "opening_line": "2",
            "closing_line": "4",
            "expression_line_count": "1",
            "blank_line_count": "0",
            "unterminated": "false",
        },
        {
            "unit_id": "b",
            "title": "",
            "opening_line": "1",
            "closing_line": "3",
            "expression_line_count": "1",
            "blank_line_count": "0",
            "unterminated": "false",
        },
        {
            "unit_id": "b",
            "title": "",
            "opening_line": "4",
            "closing_line": "6",
            "expression_line_count": "1",
            "blank_line_count": "0",
            "unterminated": "false",
        },
    ]


def test_math_block_export_write_to_path_metadata(tmp_path):
    output = tmp_path / "math-blocks.csv"

    result = export_unit_markdown_math_blocks_to_csv([{"id": "u1", "content": "$$\nx\n$$"}], output)

    assert result == {
        "path": str(output),
        "unit_count": 1,
        "rows_exported": 1,
        "bytes_written": output.stat().st_size,
    }
    assert rows(output.read_text(encoding="utf-8"))[0]["unit_id"] == "u1"
