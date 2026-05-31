import csv
from io import StringIO

from graph.export import export_unit_markdown_footnote_definition_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_footnote_definition_csv_exports_definitions_and_continuations():
    content = "Inline [^a]\n[^a]: First\n    continued\n[^b]: Second"

    assert rows(export_unit_markdown_footnote_definition_csv([{"id": "u", "title": "T", "content": content}])) == [
        {"unit_id": "u", "title": "T", "label": "a", "definition": "First continued", "line": "2", "continued_lines": "1"},
        {"unit_id": "u", "title": "T", "label": "b", "definition": "Second", "line": "4", "continued_lines": "0"},
    ]


def test_footnote_definition_csv_writes_path(tmp_path):
    path = tmp_path / "footnotes.csv"
    stats = export_unit_markdown_footnote_definition_csv([{"id": "u", "content": "[^x]: Text"}], path)

    assert rows(path.read_text(encoding="utf-8"))[0]["label"] == "x"
    assert stats["rows_exported"] == 1
