import csv
from io import StringIO

from graph.export import export_units_to_markdown_details_inventory_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_details_inventory_reports_closed_open_and_unclosed_blocks():
    content = "<details open>\n<summary>One</summary>\nbody\n</details>\n<details>\nbody"
    result = rows(export_units_to_markdown_details_inventory_csv([{"id": "u", "title": "T", "content": content}]))
    assert result == [
        {"unit_id": "u", "title": "T", "start_line": "1", "end_line": "4", "summary": "One", "is_open": "true", "line_count": "4"},
        {"unit_id": "u", "title": "T", "start_line": "5", "end_line": "6", "summary": "", "is_open": "false", "line_count": "2"},
    ]


def test_details_inventory_ignores_fenced_blocks_and_writes_path(tmp_path):
    units = [{"id": "u", "content": "```\n<details><summary>No</summary></details>\n```\n<details><summary>Yes</summary></details>"}]
    path = tmp_path / "details.csv"
    stats = export_units_to_markdown_details_inventory_csv(units, path)
    result = rows(path.read_text(encoding="utf-8"))
    assert [row["summary"] for row in result] == ["Yes"]
    assert stats["rows_exported"] == 1
