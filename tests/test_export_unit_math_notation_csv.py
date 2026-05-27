from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_math_notation_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_export_units_to_math_notation_csv_counts_inline_and_block_math():
    result = rows(
        export_units_to_math_notation_csv(
            [
                {"id": "u1", "content": "Intro $x + y$.\n$$\na^2 + b^2 = c^2\n$$\nMore $z$."},
            ]
        )
    )[0]

    assert result["unit_id"] == "u1"
    assert result["inline_math_count"] == "2"
    assert result["block_math_count"] == "1"
    assert result["unterminated_block_count"] == "0"
    assert result["first_math_line"] == "1"
    assert result["longest_math_chars"] == "15"


def test_export_units_to_math_notation_csv_counts_unterminated_blocks_and_escaped_dollars():
    result = rows(
        export_units_to_math_notation_csv(
            [
                {"id": "u1", "content": "Cost is \\$5, not math.\n$$\nopen block"},
            ]
        )
    )[0]

    assert result["inline_math_count"] == "0"
    assert result["block_math_count"] == "0"
    assert result["unterminated_block_count"] == "1"
    assert result["first_math_line"] == "2"


def test_export_units_to_math_notation_csv_sorts_rows_by_unit_id():
    result = rows(export_units_to_math_notation_csv([{"id": "b", "content": "$b$"}, {"id": "A", "content": "$a$"}]))

    assert [row["unit_id"] for row in result] == ["A", "b"]
