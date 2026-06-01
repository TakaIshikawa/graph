from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_heading_depths_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_heading_depths_exports_atx_levels_and_empty_headings():
    result = rows(
        export_unit_markdown_heading_depths_to_csv(
            [
                {
                    "id": "u1",
                    "title": "Doc",
                    "content": "# One\n## Two ##\n###\nUnderlined\n---\n```\n# Hidden\n```",
                }
            ]
        )
    )

    assert result == [
        {
            "unit_id": "u1",
            "title": "Doc",
            "line_number": "1",
            "level": "1",
            "heading_text": "One",
            "is_empty": "false",
        },
        {
            "unit_id": "u1",
            "title": "Doc",
            "line_number": "2",
            "level": "2",
            "heading_text": "Two",
            "is_empty": "false",
        },
        {
            "unit_id": "u1",
            "title": "Doc",
            "line_number": "3",
            "level": "3",
            "heading_text": "",
            "is_empty": "true",
        },
    ]


def test_markdown_heading_depths_sorts_by_unit_id_and_line_number():
    result = rows(
        export_unit_markdown_heading_depths_to_csv(
            [
                {"id": "b", "content": "text\n### Later\n# First"},
                {"id": "a", "metadata": {"title": "Meta"}, "content": "## Earlier"},
            ]
        )
    )

    assert [(row["unit_id"], row["line_number"], row["heading_text"]) for row in result] == [
        ("a", "1", "Earlier"),
        ("b", "2", "Later"),
        ("b", "3", "First"),
    ]
    assert result[0]["title"] == "Meta"


def test_markdown_heading_depths_path_mode_returns_metadata(tmp_path):
    output = tmp_path / "headings.csv"
    metadata = export_unit_markdown_heading_depths_to_csv(
        [{"id": "u1", "content": "# One"}, {"id": "u2", "content": "none"}],
        output,
    )

    assert metadata == {
        "path": str(output),
        "unit_count": 2,
        "rows_exported": 1,
        "bytes_written": output.stat().st_size,
    }
    assert rows(output.read_text(encoding="utf-8"))[0]["heading_text"] == "One"
