import csv
from io import StringIO

from graph.export import export_units_to_frontmatter_boolean_field_csv


def rows(text):
    return list(csv.DictReader(StringIO(text)))


def test_frontmatter_boolean_variants_are_normalized_with_locations():
    data = rows(export_units_to_frontmatter_boolean_field_csv([{"id": "u1", "title": "T", "content": "---\npublished: TRUE\nhidden: false\n---\nBody"}]))

    assert data == [
        {"unit_id": "u1", "title": "T", "key_path": "hidden", "value": "false", "original_token": "false", "line_number": "3"},
        {"unit_id": "u1", "title": "T", "key_path": "published", "value": "true", "original_token": "TRUE", "line_number": "2"},
    ]


def test_frontmatter_boolean_export_uses_yaml_resolution_and_nested_paths():
    data = rows(export_units_to_frontmatter_boolean_field_csv([{"id": "u1", "content": "---\nreview:\n  approved: off\n  note: yes please\n---"}]))

    assert data == [{"unit_id": "u1", "title": "", "key_path": "review.approved", "value": "false", "original_token": "off", "line_number": "3"}]
