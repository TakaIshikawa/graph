import csv
from io import StringIO

from graph.export import export_units_to_markdown_definition_list_csv


def _rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_markdown_definition_list_csv_exports_term_definition_pairs():
    content = "\n".join(["Term A", ": Definition A", ": Definition B", "", "ordinary paragraph", "```", "Code", ": ignored", "```"])

    rows = _rows(export_units_to_markdown_definition_list_csv([{"id": "u1", "title": "Unit One", "source_url": "https://s.test", "content": content}]))

    assert rows == [
        {"unit_id": "u1", "title": "Unit One", "term": "Term A", "definition": "Definition A", "line_number": "2", "source_url": "https://s.test"},
        {"unit_id": "u1", "title": "Unit One", "term": "Term A", "definition": "Definition B", "line_number": "3", "source_url": "https://s.test"},
    ]


def test_markdown_definition_list_csv_path_mode_reports_write_metadata(tmp_path):
    path = tmp_path / "definitions.csv"
    units = [{"id": "u", "content": "Term\n: Definition"}]

    expected = export_units_to_markdown_definition_list_csv(units)
    stats = export_units_to_markdown_definition_list_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
