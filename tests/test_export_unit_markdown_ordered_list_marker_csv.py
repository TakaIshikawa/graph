from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_ordered_list_markers_to_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_ordered_list_marker_csv_exports_dot_paren_and_indent_outside_fences():
    text = export_unit_markdown_ordered_list_markers_to_csv(
        [
            {"id": "b", "title": "Beta", "content": "10. Ten\n  2) Two\n```\n3. Ignored\n```"},
            {"id": "a", "title": "Alpha", "content": "plain"},
        ]
    )

    assert rows(text) == [
        {"unit_id": "b", "title": "Beta", "line_number": "1", "marker": "10.", "delimiter": ".", "number": "10", "indent": "0"},
        {"unit_id": "b", "title": "Beta", "line_number": "2", "marker": "2)", "delimiter": ")", "number": "2", "indent": "2"},
    ]
