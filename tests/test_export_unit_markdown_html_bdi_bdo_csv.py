from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_unit_markdown_html_bdi_bdo_csv


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_bdi_bdo_csv_captures_dir_and_ignores_fences_empty_matches():
    content = '<bdi>name</bdi>\n<bdo dir="rtl">abc</bdo>\n<bdi></bdi>\n```\n<bdo dir="ltr">skip</bdo>\n```'

    result = rows(export_unit_markdown_html_bdi_bdo_csv([{"id": "u", "title": "T", "source": "s", "content": content}]))

    assert result == [
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "1", "tag": "bdi", "dir": "", "text": "name", "has_dir": "false"},
        {"unit_id": "u", "title": "T", "source": "s", "line_number": "2", "tag": "bdo", "dir": "rtl", "text": "abc", "has_dir": "true"},
    ]
