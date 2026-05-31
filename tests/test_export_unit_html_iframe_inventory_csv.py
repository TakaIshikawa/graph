import csv
from io import StringIO

from graph.export import export_units_to_html_iframe_inventory_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_iframe_inventory_exports_attributes_outside_fences():
    content = '<IFRAME SRC="https://a.test" title=\'Demo\' width=640 height="480" loading=lazy sandbox="allow-scripts"></iframe>\n```\n<iframe src="hidden"></iframe>\n```\n<iframe src=/embed></iframe>'

    result = rows(export_units_to_html_iframe_inventory_csv([{"id": "u", "title": "T", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "line_number": "1", "src": "https://a.test", "title_attr": "Demo", "width": "640", "height": "480", "loading": "lazy", "sandbox": "allow-scripts"},
        {"unit_id": "u", "title": "T", "line_number": "5", "src": "/embed", "title_attr": "", "width": "", "height": "", "loading": "", "sandbox": ""},
    ]


def test_iframe_inventory_sorts_by_unit_and_line():
    result = rows(export_units_to_html_iframe_inventory_csv([{"id": "b", "content": "<iframe src=b></iframe>"}, {"id": "a", "content": "\n<iframe src=a></iframe>"}]))

    assert [(row["unit_id"], row["line_number"]) for row in result] == [("a", "2"), ("b", "1")]
