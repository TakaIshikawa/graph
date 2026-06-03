from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_label_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_label_csv_for_id_nested_controls_text_and_fences():
    content = '<label for="email">Email <strong>address</strong></label>\n<label>Accept <input type="checkbox"></label>\n```\n<label>skip</label>\n```'

    result = rows(export_unit_markdown_html_label_csv([{"id": "u", "title": "T", "source": "s", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "for_id": "email", "text": "Email address", "has_control_child": "false"},
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "2", "for_id": "", "text": "Accept", "has_control_child": "true"},
    ]
