import csv
from io import StringIO

from graph.export import export_units_to_html_figure_inventory_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_figure_inventory_counts_nested_media_and_caption():
    content = '<figure>\n<a href="/x"><img src="a.png"></a><img src="b.png">\n<figcaption>A <strong>caption</strong></figcaption>\n</figure>'

    result = rows(export_units_to_html_figure_inventory_csv([{"id": "u", "title": "T", "content": content}]))

    assert result == [{"unit_id": "u", "title": "T", "line_number": "1", "has_figcaption": "True", "figcaption": "A caption", "image_count": "2", "link_count": "1"}]
