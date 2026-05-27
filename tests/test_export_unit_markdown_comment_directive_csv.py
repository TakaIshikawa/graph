import csv
from io import StringIO

from graph.export import export_units_to_markdown_comment_directive_csv


def test_comment_directive_export_parses_directives_and_ignores_noise():
    text = "\n".join(["<!-- TODO: ship it -->", "<!-- ordinary prose -->", "```", "<!-- FIXME: hidden -->", "```", "<!-- @owner taka -->"])
    rows = list(csv.DictReader(StringIO(export_units_to_markdown_comment_directive_csv([{"id": "u1", "title": "T", "content": text}]))))

    assert rows == [
        {"unit_id": "u1", "title": "T", "line_number": "1", "directive": "todo", "payload": "ship it", "raw_comment": "<!-- TODO: ship it -->"},
        {"unit_id": "u1", "title": "T", "line_number": "6", "directive": "@owner", "payload": "taka", "raw_comment": "<!-- @owner taka -->"},
    ]
