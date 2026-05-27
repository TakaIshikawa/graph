import csv
from io import StringIO

from graph.export import export_units_to_markdown_superscript_inventory_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_superscript_inventory_sorts_and_ignores_fences():
    text = "Line ^z^\n```md\n^skip^\n```\nAgain ^alpha^ and ^b^"
    result = rows(export_units_to_markdown_superscript_inventory_csv([{"id": "b", "content": "^c^"}, {"id": "a", "title": "A", "content": text}]))
    assert [(row["unit_id"], row["line_number"], row["text"], row["character_count"]) for row in result] == [
        ("a", "1", "z", "1"),
        ("a", "5", "alpha", "5"),
        ("a", "5", "b", "1"),
        ("b", "1", "c", "1"),
    ]
    assert result[0]["marker"] == "^z^"
    assert result[0]["title"] == "A"


def test_superscript_inventory_writes_path(tmp_path):
    path = tmp_path / "superscript.csv"
    expected = export_units_to_markdown_superscript_inventory_csv([{"id": "u", "content": "^x^"}])
    stats = export_units_to_markdown_superscript_inventory_csv([{"id": "u", "content": "^x^"}], path)
    assert path.read_text(encoding="utf-8") == expected
    assert stats["rows_exported"] == 1
