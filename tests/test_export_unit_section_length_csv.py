from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_section_length_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_section_length_ignores_fenced_headings_and_formats_average(tmp_path):
    text = "Intro\n# One\na\nb\n```\n# ignored\n```\n## Two\nc"
    output = tmp_path / "sections.csv"
    result = export_units_to_section_length_csv([{"id": "u", "content": text}], output)
    row = rows(output.read_text(encoding="utf-8"))[0]

    assert result["unit_count"] == 1
    assert row["heading_count"] == "2"
    assert row["section_count"] == "3"
    assert row["max_section_line_count"] == "5"
    assert row["average_section_line_count"] == "2.33"
    assert row["longest_section_heading"] == "One"


def test_section_length_content_without_headings_is_one_section():
    row = rows(export_units_to_section_length_csv([{"id": "u", "content": "a\nb"}]))[0]

    assert row["heading_count"] == "0"
    assert row["section_count"] == "1"
