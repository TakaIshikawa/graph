import csv
from io import StringIO

from graph.export import export_units_to_markdown_link_definition_title_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_link_definition_title_exports_supported_title_styles():
    content = '[a]: https://a "Alpha"\n[b]: /b \'Beta\'\n[c]: /c (Gamma)\n[d]: /d\n```\n[x]: /x "Hidden"\n```'

    result = rows(export_units_to_markdown_link_definition_title_csv([{"id": "u", "title": "T", "content": content}]))

    assert [(row["label"], row["url"], row["link_title"], row["title_style"], row["line_number"]) for row in result] == [
        ("a", "https://a", "Alpha", "quoted", "1"),
        ("b", "/b", "Beta", "single_quoted", "2"),
        ("c", "/c", "Gamma", "parenthesized", "3"),
    ]
