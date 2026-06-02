import csv
from io import StringIO

from graph.export import export_unit_markdown_html_lang_attributes_to_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_html_lang_attribute_csv_exports_normalized_lang_and_ignores_fences():
    content = '<p lang="EN-us">Hello</p>\n```html\n<span lang="fr">Non</span>\n```\n<div lang=ja>Hi</div>'

    result = rows(export_unit_markdown_html_lang_attributes_to_csv([{"id": "u", "title": "T", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "line_number": "1", "tag_name": "p", "lang": "en-us", "raw_snippet": '<p lang="EN-us">'},
        {"unit_id": "u", "title": "T", "line_number": "5", "tag_name": "div", "lang": "ja", "raw_snippet": "<div lang=ja>"},
    ]
