import csv
from io import StringIO

from graph.export import export_unit_markdown_link_download_attributes_to_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_link_download_attribute_csv_exports_bare_and_named_downloads():
    content = '<a href="/a.pdf" download>PDF</a>\n```html\n<a href="/skip" download></a>\n```\n<a download="file.zip" href="/b">Zip</a>'

    result = rows(export_unit_markdown_link_download_attributes_to_csv([{"id": "u", "title": "T", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "line_number": "1", "href": "/a.pdf", "download_filename": "", "raw_snippet": '<a href="/a.pdf" download>'},
        {"unit_id": "u", "title": "T", "line_number": "5", "href": "/b", "download_filename": "file.zip", "raw_snippet": '<a download="file.zip" href="/b">'},
    ]


def test_link_download_attribute_csv_writes_path(tmp_path):
    path = tmp_path / "downloads.csv"
    stats = export_unit_markdown_link_download_attributes_to_csv([{"id": "u", "content": '<a href="/x" download="x.txt">'}], path)

    assert stats["rows_exported"] == 1
    assert rows(path.read_text(encoding="utf-8"))[0]["download_filename"] == "x.txt"
