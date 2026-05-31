import csv
from io import StringIO

from graph.export import export_units_to_markdown_collapsed_section_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_collapsed_section_csv_detects_details_and_callouts_outside_fences():
    content = '<details open><summary>More <em>info</em></summary>\n</details>\n> [!note]- Label\n```\n<details><summary>Hidden</summary>\n> [!tip]- Hidden\n```'

    result = rows(export_units_to_markdown_collapsed_section_csv([{"id": "u", "title": "T", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "line_number": "1", "section_type": "details", "label": "More info", "starts_open": "True"},
        {"unit_id": "u", "title": "T", "line_number": "3", "section_type": "obsidian_callout", "label": "Label", "starts_open": "False"},
    ]
