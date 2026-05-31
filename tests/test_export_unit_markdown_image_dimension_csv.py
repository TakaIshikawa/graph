import csv
from io import StringIO

from graph.export import export_units_to_markdown_image_dimension_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_image_dimension_csv_detects_attribute_and_url_dimensions():
    content = '![A](a.png){width=640 height=480}\n![B](b.png?width=300&height=200)\n![C](c.png#width=10&height=20)\n![D](d.png)\n```\n![X](x.png?width=1)\n```'

    result = rows(export_units_to_markdown_image_dimension_csv([{"id": "u", "title": "T", "content": content}]))

    assert [(row["alt_text"], row["target"], row["width"], row["height"], row["dimension_source"]) for row in result] == [
        ("A", "a.png", "640", "480", "attribute"),
        ("B", "b.png?width=300&height=200", "300", "200", "url"),
        ("C", "c.png#width=10&height=20", "10", "20", "url"),
    ]
