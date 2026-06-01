from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_thematic_breaks_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_markdown_thematic_breaks_exports_valid_breaks():
    result = rows(
        export_unit_markdown_thematic_breaks_to_csv(
            [{"id": "u1", "title": "Doc", "content": "***\n- - -\n____"}]
        )
    )

    assert result == [
        {"unit_id": "u1", "title": "Doc", "line_number": "1", "marker_style": "*", "normalized_marker_length": "3"},
        {"unit_id": "u1", "title": "Doc", "line_number": "2", "marker_style": "-", "normalized_marker_length": "3"},
        {"unit_id": "u1", "title": "Doc", "line_number": "3", "marker_style": "_", "normalized_marker_length": "4"},
    ]


def test_markdown_thematic_breaks_ignores_short_invalid_markers():
    result = rows(export_unit_markdown_thematic_breaks_to_csv([{"id": "u1", "content": "**\n--\n__"}]))

    assert result == []


def test_markdown_thematic_breaks_ignores_markers_inside_fenced_code_blocks():
    result = rows(export_unit_markdown_thematic_breaks_to_csv([{"id": "u1", "content": "```\n***\n```\n___"}]))

    assert result == [
        {"unit_id": "u1", "title": "", "line_number": "4", "marker_style": "_", "normalized_marker_length": "3"}
    ]
