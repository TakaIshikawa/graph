import csv
from io import StringIO

from graph.export import export_units_to_html_microdata_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_html_microdata_export_finds_attributes_and_boolean_itemscope():
    content = '<div itemscope itemtype="https://schema.org/Book"><span itemprop=\'name\'>T</span></div>\n```\n<p itemprop="skip">\n```'
    result = rows(export_units_to_html_microdata_csv([{"id": "u", "title": "T", "content": content}]))
    assert [(row["tag_name"], row["attribute"], row["value"], row["line_number"]) for row in result] == [
        ("span", "itemprop", "name", "1"),
        ("div", "itemscope", "", "1"),
        ("div", "itemtype", "https://schema.org/Book", "1"),
    ]


def test_html_microdata_export_writes_path(tmp_path):
    path = tmp_path / "microdata.csv"
    stats = export_units_to_html_microdata_csv([{"id": "u", "content": "<a itemid=x itemref='y'>"}], path)
    assert [row["attribute"] for row in rows(path.read_text(encoding="utf-8"))] == ["itemid", "itemref"]
    assert stats["rows_exported"] == 2
