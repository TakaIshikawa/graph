import csv
from io import StringIO

from graph.export import export_units_to_markdown_horizontal_rule_csv


def test_horizontal_rule_export_skips_frontmatter_and_fences():
    content = "---\ntitle: T\n---\n***\n- - -\n```\n___\n```"
    rows = list(csv.DictReader(StringIO(export_units_to_markdown_horizontal_rule_csv([{"id": "u1", "content": content}]))))

    assert rows == [
        {"unit_id": "u1", "title": "", "line_number": "4", "marker_character": "*", "marker_count": "3", "raw_line": "***"},
        {"unit_id": "u1", "title": "", "line_number": "5", "marker_character": "-", "marker_count": "3", "raw_line": "- - -"},
    ]
