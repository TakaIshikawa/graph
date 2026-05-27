from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_math_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_math_inventory_counts_display_inline_and_unterminated(tmp_path):
    text = "Price $10 but math $x+1$.\n$$\na\nb\n$$\n$$\nc"
    output = tmp_path / "math.csv"
    result = export_units_to_math_inventory_csv([{"id": "u", "content": text}], output)
    row = rows(output.read_text(encoding="utf-8"))[0]

    assert result["bytes_written"] == output.stat().st_size
    assert row["display_math_block_count"] == "1"
    assert row["inline_math_span_count"] == "1"
    assert row["unterminated_display_math_count"] == "1"
    assert row["max_display_math_line_count"] == "2"
