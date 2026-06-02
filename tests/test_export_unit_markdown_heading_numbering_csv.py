from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_heading_numbering_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_atx_and_setext_numbered_headings():
    content = "# 1. Overview\n## 2.3 Details\nIV. Notes\n--------\n- 1. list item"

    result = rows(export_units_to_markdown_heading_numbering_csv([{"id": "u", "title": "T", "source": "s", "content": content}]))

    assert [(row["line_number"], row["heading_depth"], row["numbering_style"], row["raw_number"], row["normalized_number"], row["heading_text"]) for row in result] == [
        ("1", "1", "numeric", "1.", "1", "Overview"),
        ("2", "2", "dotted", "2.3", "2.3", "Details"),
        ("3", "2", "roman", "IV.", "iv", "Notes"),
    ]
    assert result[0]["source"] == "s"


def test_ignores_ordered_lists_and_fenced_code():
    content = "1. Not a heading\n```md\n# 1. Ignored\n```\nA. Appendix\n="

    result = rows(export_units_to_markdown_heading_numbering_csv([{"id": "u", "content": content}]))

    assert [(row["numbering_style"], row["heading_text"]) for row in result] == [("alphabetic", "Appendix")]


def test_path_write_returns_export_metadata(tmp_path):
    output = tmp_path / "heading_numbering.csv"

    result = export_units_to_markdown_heading_numbering_csv([{"id": "u", "content": "# 1. A"}], output)

    assert result == {"path": str(output), "unit_count": 1, "rows_exported": 1, "bytes_written": output.stat().st_size}
