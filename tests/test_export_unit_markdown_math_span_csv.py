from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_math_span_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_inline_math_spans():
    result = rows(export_units_to_markdown_math_span_csv([{"id": "u", "title": "T", "source": "s", "content": "Use $x + y$ and $z$."}]))

    assert [(row["expression"], row["character_length"], row["delimiter_style"], row["position"]) for row in result] == [
        ("x + y", "5", "$", "5"),
        ("z", "1", "$", "17"),
    ]
    assert result[0]["source"] == "s"


def test_ignores_block_math_escaped_currency_code_and_fences():
    content = r"Price is $5 and escaped \$x$ and `$a$`" + "\n$$x$$\n```md\n$a$\n```\nKeep $b$"

    result = rows(export_units_to_markdown_math_span_csv([{"id": "u", "content": content}]))

    assert [(row["expression"], row["line_number"]) for row in result] == [("b", "6")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "math.csv"

    result = export_units_to_markdown_math_span_csv([{"id": "u", "content": "$x$"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
