import csv
from io import StringIO

from graph.export import export_units_to_html_aria_attribute_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_html_aria_export_handles_multiple_quotes_and_case():
    content = '<button ARIA-label="Save" aria-expanded=\'false\' aria-live=polite>Save</button>'
    result = rows(export_units_to_html_aria_attribute_csv([{"id": "u", "title": "T", "content": content}]))
    assert [(row["tag"], row["attribute"], row["value"]) for row in result] == [
        ("button", "aria-expanded", "false"),
        ("button", "aria-label", "Save"),
        ("button", "aria-live", "polite"),
    ]


def test_html_aria_export_no_match_and_path(tmp_path):
    assert export_units_to_html_aria_attribute_csv([{"id": "u", "content": "<p>No aria</p>"}]) == "unit_id,title,tag,attribute,value,line_number\n"
    path = tmp_path / "aria.csv"
    stats = export_units_to_html_aria_attribute_csv([{"id": "u", "content": "<div aria-hidden=true>"}], path)
    assert rows(path.read_text(encoding="utf-8"))[0]["value"] == "true"
    assert stats["rows_exported"] == 1
