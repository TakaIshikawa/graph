import csv
from io import StringIO

from graph.export import export_units_to_markdown_abbreviation_inventory_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_markdown_abbreviation_inventory_exports_rows_outside_fences():
    content = "\n".join(
        [
            "*[HTML]: HyperText Markup Language",
            "```md",
            "*[CSS]: Cascading Style Sheets",
            "```",
            "  *[ARIA]:  Accessible Rich Internet Applications  ",
        ]
    )

    assert rows(export_units_to_markdown_abbreviation_inventory_csv([{"id": "u", "title": "T", "content": content}])) == [
        {
            "unit_id": "u",
            "title": "T",
            "line_number": "1",
            "abbreviation": "HTML",
            "definition": "HyperText Markup Language",
        },
        {
            "unit_id": "u",
            "title": "T",
            "line_number": "5",
            "abbreviation": "ARIA",
            "definition": "Accessible Rich Internet Applications",
        },
    ]


def test_markdown_abbreviation_inventory_sorts_by_unit_line_and_abbreviation():
    units = [
        {"id": "b", "content": "*[B]: Beta"},
        {"id": "a", "content": "*[Z]: Zed\n*[A]: Alpha"},
    ]

    result = rows(export_units_to_markdown_abbreviation_inventory_csv(units))

    assert [(row["unit_id"], row["line_number"], row["abbreviation"]) for row in result] == [("a", "1", "Z"), ("a", "2", "A"), ("b", "1", "B")]


def test_markdown_abbreviation_inventory_writes_path(tmp_path):
    path = tmp_path / "abbr.csv"

    stats = export_units_to_markdown_abbreviation_inventory_csv([{"id": "u", "content": "*[API]: Application Programming Interface"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["definition"] == "Application Programming Interface"
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] > 0
