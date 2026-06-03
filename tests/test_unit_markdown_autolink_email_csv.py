from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_units_to_markdown_autolink_email_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_exports_angle_bracket_email_autolinks_only():
    result = rows(export_units_to_markdown_autolink_email_csv([{"id": "u", "content": "Raw a@example.com, link <https://example.com>, mail <Name@Example.COM>."}]))

    assert result == [{"unit_id": "u", "title": "", "email": "Name@Example.COM", "domain": "example.com", "line_number": "1", "excerpt": "Raw a@example.com, link <https://example.com>, mail <Name@Example.COM>."}]
