import csv
from io import StringIO

from graph.export import export_units_to_markdown_link_title_csv


def _rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_markdown_link_title_csv_captures_quoted_inline_link_titles():
    content = "\n".join(
        [
            '[Alpha](https://a.test "Title A") and [Beta](beta.md \'Title B\')',
            "[Plain](https://plain.test)",
            '![Image](image.png "Image title")',
            "```",
            '[Ignored](https://ignored.test "No")',
            "```",
        ]
    )

    rows = _rows(export_units_to_markdown_link_title_csv([{"id": "u1", "title": "Unit One", "content": content}]))

    assert rows == [
        {"unit_id": "u1", "title": "Unit One", "line_number": "1", "link_text": "Beta", "target": "beta.md", "title_text": "Title B"},
        {"unit_id": "u1", "title": "Unit One", "line_number": "1", "link_text": "Alpha", "target": "https://a.test", "title_text": "Title A"},
    ]


def test_markdown_link_title_csv_path_mode_reports_write_metadata(tmp_path):
    path = tmp_path / "link_titles.csv"
    units = [{"id": "u", "content": '[Doc](doc.md "Documentation")'}]

    expected = export_units_to_markdown_link_title_csv(units)
    stats = export_units_to_markdown_link_title_csv(units, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats["unit_count"] == 1
    assert stats["rows_exported"] == 1
    assert stats["bytes_written"] == path.stat().st_size
