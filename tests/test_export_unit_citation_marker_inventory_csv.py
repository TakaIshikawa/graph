from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_citation_marker_inventory_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_unit_citation_marker_inventory_classifies_markers_in_order():
    text = export_units_to_citation_marker_inventory_csv(
        [{"id": "a", "title": "Alpha", "content": "Claim [1] (Smith, 2020)\nDOI 10.1000/XYZ.9 and note [^n1]"}]
    )

    assert [(row["marker_type"], row["marker_text"], row["line_number"], row["occurrence_index"]) for row in rows(text)] == [
        ("numeric_bracket", "[1]", "1", "1"),
        ("author_year", "(Smith, 2020)", "1", "2"),
        ("doi", "10.1000/XYZ.9", "2", "3"),
        ("footnote_ref", "[^n1]", "2", "4"),
    ]


def test_unit_citation_marker_inventory_empty_input_returns_header():
    assert export_units_to_citation_marker_inventory_csv([]) == "unit_id,title,marker_type,marker_text,line_number,occurrence_index\n"
