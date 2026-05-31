import csv
from io import StringIO

from graph.export import export_units_to_frontmatter_reference_field_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_frontmatter_reference_field_csv_flattens_and_classifies_values():
    content = "---\nurl: https://example.com\ndoi: 10.1000/xyz123\nrefs:\n  - '@smith2020'\n  - '[[Note]]'\n  - unit-123\n---\nBody"

    result = rows(export_units_to_frontmatter_reference_field_csv([{"id": "u", "title": "T", "content": content}]))

    assert [(row["field_path"], row["reference_type"], row["value"]) for row in result] == [
        ("doi", "doi", "10.1000/xyz123"),
        ("refs[0]", "citekey", "@smith2020"),
        ("refs[1]", "wikilink", "[[Note]]"),
        ("refs[2]", "unit_id_reference", "unit-123"),
        ("url", "url", "https://example.com"),
    ]
