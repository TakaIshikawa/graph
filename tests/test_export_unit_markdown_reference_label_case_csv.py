import csv
from io import StringIO

from graph.export import export_units_to_markdown_reference_label_case_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_reference_label_case_groups_definitions_and_references_case_insensitively():
    content = "[text][Label]\n[other][label]\n[Label]: https://example.com\n```md\n[Skip]: https://example.com\n[x][SKIP]\n```"
    result = rows(export_units_to_markdown_reference_label_case_csv([{"id": "u", "title": "T", "content": content}]))
    assert result == [
        {
            "unit_id": "u",
            "title": "T",
            "normalized_label": "label",
            "observed_labels": "Label|label",
            "definition_count": "1",
            "reference_count": "2",
            "has_case_conflict": "true",
        }
    ]


def test_reference_label_case_writes_path(tmp_path):
    path = tmp_path / "refs.csv"
    stats = export_units_to_markdown_reference_label_case_csv([{"id": "u", "content": "[x][A]\n[a]: /a"}], path)
    assert rows(path.read_text(encoding="utf-8"))[0]["has_case_conflict"] == "true"
    assert stats["rows_exported"] == 1
